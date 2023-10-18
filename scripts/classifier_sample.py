"""
Generate samples from a trained classifier.
Inputs:
- ClassifierClass
- 

"""


import wandb # Experiment tracking
from classifiers import get_classifiers
import argparse
import os, time, gc

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from math import sqrt
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from scripts.imagenet_sample import load_selected_imagenet_samples, load_imagenet_class, get_imagenet_class_name, unload_imagenet_dataset

def check_must_split_classifier(args):
    if args.classifier_class == "UnetEncoderClassifier":
        return True

    if args.classifier_class == "ConvNeXtLClassifier" and args.cone_projection:
        return True

    return False
    
def clear_cuda_cache(out_delay=2, in_delay=0, log=True):
    if log:
        logger.log(f"!!!! Clearing CUDA cache with {out_delay}s grace period")

    time.sleep(in_delay)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    th.cuda.empty_cache()
    gc.collect()
    time.sleep(out_delay)
    
def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = create_argparser().parse_args()
    #args.seed = 1
    args.model_output_size = 256

    assert args.classifier_class is not None, "Must specify classifier class"
    
    
    if check_must_split_classifier(args):
        # Too big for memory, so we have to split it: main model on 0/1/2 and UNET on 3
        logger.log("!!!! Splitting model and classifier(s) to different GPUs !!!!")
        dist_util.set_classifier_device("cuda:3", "3")
        dist_util.set_default_device("cuda:0", "0,1,2")
    

    set_seed(args.seed)

    model_config = model_and_diffusion_defaults()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="dvces_rev5",
        #entity="fhws_cairo",
        # track args and model config 
        config= vars(args) | model_config
    )
    
    # generate an unique name for this run
    wandb.run.name = f"{args.classifier_class}_xzero{1 if args.xzero_prediction else 0}_cone{1 if args.cone_projection else 0}_t{int(time.time())}"
    
    # define a metric we are interested in the minimum of
    wandb.define_metric("target_class_conf_std", summary="min")
    # define a metric we are interested in the maximum of
    wandb.define_metric("target_class_conf_mean", summary="max")
    
    wandb.define_metric("original_class_conf_mean", summary="min")
    wandb.define_metric("original_class_conf_std", summary="min")

    # 
    # define our custom x axis metric
    wandb.define_metric("timestep")
    # set all other train/ metrics to use this step
    wandb.define_metric("batch_mean_acc", step_metric="timestep")


    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval().to(dist_util.dev())
            
    if args.use_fp16:
        model.convert_to_fp16()

    model = th.nn.DataParallel(model)

    logger.log("loading classifier...")

    # Load the classifier and create an instance
    classifier_cls = getattr(__import__('classifiers', fromlist=[args.classifier_class]), args.classifier_class)
    classifier = classifier_cls()
    classifier.init_model(args)
    classifier.load(args.classifier_path)
    classifier.perturbate(args.classifier_perturbation)
    classifier.model.to(dist_util.dev_classifier())
    if args.classifier_use_fp16: # warning: not tested!
        classifier.convert_to_fp16()
    classifier.model.eval()

    if args.cone_projection:
        robust_classifier_cls = getattr(__import__('classifiers', fromlist=["MadryClassifier"]), "MadryClassifier")
        robust_classifier = robust_classifier_cls()
        robust_classifier.init_model(args)
        robust_classifier.load(args.classifier_path)
        robust_classifier.model.to(dist_util.dev_classifier())
        if args.classifier_use_fp16:
            robust_classifier.convert_to_fp16()
        robust_classifier.model.eval()
        
    # FROM DVCE
    def cone_projection(grad_temp_1, grad_temp_2, deg=30):
        angles_before = th.acos(
            (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))

        grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
            grad_temp_1.shape[0], -1) * grad_temp_2
        grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        radians = th.tensor([deg], device=grad_temp_1.device).deg2rad()

        cone_projection = grad_temp_1 * th.tan(radians) + grad_temp_2

        # second classifier is a non-robust one -
        # unless we are less than 45 degrees away - don't cone project
        grad_temp = grad_temp_2.clone()
        grad_temp[angles_before > radians] = cone_projection[angles_before > radians]
        return grad_temp

    def l2_normalize_gradient(grad, small_const=1e-22):
        grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
        grad_norm = th.where(grad_norm < small_const, grad_norm+small_const, grad_norm)
        gradient_normalized = grad / grad_norm
        return gradient_normalized

    def cond_fn(x, t, pred_xstart, init_image_data=None, y=None):
        assert y is not None
        if not args.xzero_prediction:
            x = pred_xstart
                
        with th.no_grad():
            with th.enable_grad():
                classifer_logits = classifier.call(pred_xstart.to(dist_util.dev_classifier()), t.to(dist_util.dev_classifier()), preprocessing=False)
                classifier_log_probs = F.log_softmax(classifer_logits, dim=-1)
                classifier_log_probs_selected = classifier_log_probs[range(args.batch_size), y.view(-1).to(dist_util.dev_classifier())]
                target_grad = th.autograd.grad(classifier_log_probs_selected.mean(), x, retain_graph=True)[0]
                            
                if args.cone_projection:
                    robust_logits = robust_classifier.call(pred_xstart.to(dist_util.dev_classifier()), t.to(dist_util.dev_classifier()), preprocessing=False)
                    robust_log_probs = F.log_softmax(robust_logits, dim=-1)
                    robust_log_probs_selected = robust_log_probs[range(args.batch_size), y.view(-1).to(dist_util.dev_classifier())]
                    

                if args.cone_projection:
                    robust_classifier_grad = th.autograd.grad(robust_log_probs_selected.mean(), x, retain_graph=True)[0]
                    
                    target_grad = cone_projection(robust_classifier_grad.view(x.shape[0], -1),
                                                            target_grad.view(x.shape[0], -1),
                                                            30).view_as(robust_classifier_grad)
                
            target_grad_normalized = l2_normalize_gradient(target_grad)
            classifier_gradient = target_grad_normalized * args.classifier_scale

        with th.no_grad():
            if init_image_data is not None:
                diff = pred_xstart - init_image_data
                lp_dist = (args.lp_custom * diff.abs() ** (args.lp_custom - 1)) * diff.sign()
                lp_regularization_normalized = l2_normalize_gradient(lp_dist)
                lp_regularization = args.lp_regularization_scale * lp_regularization_normalized
            else:
                lp_regularization = 0

            return classifier_gradient - lp_regularization

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    image_classes = [293, 209, 935, 973, 970, 965]
    target_classes = [288, 292, 207, 208, 924, 959, 970, 972, 980, 963, 965]


    all_metrics = {
        "target_class_conf_std": [],
        "target_class_conf_mean": [],
        "original_class_conf_std": [],
        "original_class_conf_mean": [],
        "lp_norm_1": [],
        "lp_norm_1_5": [],
        "lp_norm_2": [],
        "original_class_validity": [],
        "target_class_validity": [],
        "fid": [],
        "lpips": [],
    }
    

    single_metrics_table = []

    # We need to save the images for the ORACLE calculation
    result_images = []
    result_image_grids = [] # for the large table

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=64).to(dist_util.dev())

    for image_cls in image_classes:
        metrics = {
            "target_class_conf_std": [],
            "target_class_conf_mean": [],
            "original_class_conf_std": [],
            "original_class_conf_mean": [],
            "lp_norm_1": [],
            "lp_norm_1_5": [],
            "lp_norm_2": [],
            "original_class_validity": [],
            "target_class_validity": [],
            "fid": [],
            "lpips": [],    
        }
        row_result_images = []
        row_result_image_grids = []
        result_image_grids.append(row_result_image_grids)
        result_images.append(row_result_images)
        for label_cls in target_classes:
            names = get_imagenet_class_name(args.data_path, [image_cls, label_cls])
            run_id = f"run_{names[0]}({image_cls})_to_{names[1]}({label_cls})".replace(" ", "-")
            print("run id: ", run_id)
            init_image_data = load_imagenet_class(args.data_path,args.model_output_size, image_cls, label_cls)


            classes = init_image_data["targets"].to(dist_util.dev())
            init_images = init_image_data["images"].to(dist_util.dev())

            # reset the seed for every run
            set_seed(args.seed)
            sample = diffusion.p_sample_loop(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs={"y":classes},
                cond_fn=cond_fn,
                device=dist_util.dev(),
                skip_timesteps=args.skip_timesteps, # Added to be able to skip the conditioning for timesteps #int(args.timestep_respacing)
                init_image_data=init_images if args.init_image else None, # Loaded images that should be defused
            )

            # Save imgs
            n_unique_imgs = len(init_image_data["images"].unique(dim=0))
            n_row = len(init_image_data["images"])//n_unique_imgs
            imgs_and_samples = th.cat(
                (init_images[::n_row][:,None], sample.view(n_unique_imgs,n_row,3,args.image_size,args.image_size)),dim=1) \
                .view(-1,3,args.image_size,args.image_size)
            all_labels = th.cat((init_image_data["labels"][::n_row][:,None], init_image_data["targets"].view(n_unique_imgs,n_row)),dim=1).view(-1)
            img_grid = torchvision.utils.make_grid(imgs_and_samples, nrow=imgs_and_samples.shape[0]//n_unique_imgs).permute(1, 2, 0)


            with th.no_grad():
                # Compute probabilities
                final_timestep = th.zeros((args.batch_size,),device=dist_util.dev())
                logits = classifier.call(sample.to(dist_util.dev_classifier()), final_timestep.to(dist_util.dev_classifier()))
                logits = logits.to(dist_util.dev()) # back to main device
                probs = F.softmax(logits, dim=-1)
                target_class_probs = probs[range(len(logits)), classes.view(-1)]
                target_class_conf_std, target_class_conf_mean = th.std_mean(target_class_probs)
                original_class_probs = probs[range(len(logits)), init_image_data["labels"].view(-1).cuda()]
                original_class_conf_std, original_class_conf_mean = th.std_mean(original_class_probs)
                
                # Log metrics and samples

                metrics["target_class_conf_std"].append(target_class_conf_std.item())
                metrics["target_class_conf_mean"].append(target_class_conf_mean.item())

                metrics["original_class_conf_std"].append(original_class_conf_std.item())
                metrics["original_class_conf_mean"].append(original_class_conf_mean.item())
                
                # Metrics
                ################################################
                # Implement FID                
                init_images_uint8_cuda = (init_images*255).type(th.uint8).cuda().to(dist_util.dev())
                sample_uint8_cuda = (sample*255).type(th.uint8).cuda().to(dist_util.dev())                
                
                fid.update(init_images_uint8_cuda, real=True)
                fid.update(sample_uint8_cuda, real=False)
                fid_val = fid.compute().item()
                metrics["fid"].append(fid_val)
                fid.reset() # reset for next run

                ################################################
                # Implement Flip Ratio / Target Class Validity - how many samples correspond to the target class
                flr = th.argmax(logits, dim=1) == classes.view(-1)
                target_class_validity = th.mean(flr.float()).item()
                metrics["target_class_validity"].append(target_class_validity)
                
                ################################################
                # Implement Flip Out Ratio / Original Class Validity
                #  the same as flip ratio but input class, i.e., how many samples remained in the original class
                flr = th.argmax(logits, dim=1) == init_image_data["labels"].view(-1).cuda()
                original_class_validity = th.mean(flr.float()).item()
                metrics["original_class_validity"].append(original_class_validity)
                
                ################################################
                # Implement CLOSENESS / lp norm
                # lp norm: just a distance between the original image and the final image
                # calculate for p = 1, 1.5, 2
                # average it over the 50 images
                p = [1, 1.5, 2]
                lp_norms = []
                for i in p:
                    lp_norm = th.norm(sample - init_images, p=i)
                    lp_norm=th.mean(lp_norm).item()
                    lp_norms.append(lp_norm)
                    id = str(i).replace(".", "_")
                    metrics[f"lp_norm_{id}"].append(lp_norm)
                
                
                ################################################
                # Implement CLOSENESS / LPIPS
                # use torchmetrics for this
                # https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
                # use "alex" for all of them
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(dist_util.dev())
                
                # normalize images to [-1,1]
                sample_norm = ((sample - sample.min()) / (sample.max() - sample.min())) * 2 - 1
                init_images_norm = ((init_images - init_images.min()) / (init_images.max() - init_images.min())) * 2 - 1
                
                lpips_score = lpips(sample_norm, init_images_norm).item()
                metrics[f"lpips"].append( lpips_score)
                
                

                images = wandb.Image(
                    img_grid.cpu().numpy(),
                    caption=f"{run_id} | Classifier_Acc: {target_class_conf_mean.item()*100}"
                )
                row_result_image_grids.append(images)
                wandb.log({
                    f"target_class_conf_std": target_class_conf_std.item(),
                    f"target_class_conf_mean": target_class_conf_mean.item(),
                    f"original_class_conf_std": original_class_conf_std.item(),
                    f"original_class_conf_mean": original_class_conf_mean.item(),
                    f"fid": fid_val,
                    f"target_class_validity": target_class_validity,
                    f"original_class_validity": original_class_validity,
                    f"lp_norm_1": lp_norms[0],
                    f"lp_norm_1.5": lp_norms[1],
                    f"lp_norm_2": lp_norms[2],
                    f"lpips": lpips_score,
                    f"run_id": run_id,
                    f"images": images,
                    f"{run_id}": images,
                })

                

                single_metrics_table.append(dict(
                    run_id=run_id,
                    # hyperparameters
                    classifier=args.classifier_class,
                    cone_projection=args.cone_projection,
                    xzero_prediction=args.xzero_prediction,

                    # source and target
                    source_class=image_cls,
                    source_class_name=names[0],
                    target_class=label_cls,
                    target_class_name=names[1],


                    # metrics
                    target_class_conf_std=target_class_conf_std.item(),
                    target_class_conf_mean=target_class_conf_mean.item(),
                    original_class_conf_std=original_class_conf_std.item(),
                    original_class_conf_mean=original_class_conf_mean.item(),
                    fid=fid_val,
                    target_class_validity=target_class_validity,
                    original_class_validity=original_class_validity,
                    lp_norm_1=lp_norms[0],
                    lp_norm_1_5=lp_norms[1],
                    lp_norm_2=lp_norms[2],
                    lpips=lpips_score
                ))

                # ensure directory and save images
                os.makedirs(f"{wandb.run.dir}/raw_images", exist_ok=True)
                np.savez(f"{wandb.run.dir}/raw_images/{run_id}_all_imgs_labels.npz", imgs_and_samples.cpu(), all_labels.cpu())
                
                # Save imgs in result_images
                row_result_images.append(sample.cpu().detach())

                
            del sample
            del logits
            del init_image_data
            del init_images
            clear_cuda_cache(out_delay=0, in_delay=0, log=False)

        for metric in metrics:
            all_metrics[metric].append(metrics[metric])

    import pandas as pd

    # map class to name
                              

    for k in all_metrics:
        df = pd.DataFrame(all_metrics[k],
                          columns=get_imagenet_class_name(args.data_path, target_classes), 
                          index=get_imagenet_class_name(args.data_path, image_classes))
        df.index.name = 'source_class'
        df = df.reset_index()        
        table = wandb.Table(dataframe=df)
        wandb.log({f"metric_{k}": table})
        df.to_csv(f"{wandb.run.dir}/metric_{k}.csv")


    ################################################################
    # ORACLE SCORES
    ################################################################
    # unload the model because we need the memory
    logger.log("Oracle: unload model")
    del model
    unload_imagenet_dataset()


    clear_cuda_cache(out_delay=2, in_delay=2)
    
    logger.log("Oracle: init classifiers")
    all_classifiers = get_classifiers()
    for i, oracle_classifier in enumerate(all_classifiers):
        logger.log(f"Oracle: init classifier {i}={oracle_classifier.name}")
        if oracle_classifier.name == classifier.name:
            # override the classifier we used for sampling as it is already loaded
            all_classifiers[i] = classifier
            classifier.model.to(dist_util.dev())
            continue 
        
        # init all classifiers
        oracle_classifier.init_model(args)
        oracle_classifier.load(args.classifier_path)
        oracle_classifier.perturbate(args.classifier_perturbation)
        oracle_classifier.model.to(dist_util.dev())
        if args.classifier_use_fp16:
            oracle_classifier.convert_to_fp16()
        #classifier.enable_dataparallel()
        oracle_classifier.model.eval()
        
    logger.log("Oracle: calculate scores")
    oracle_scores = []
    timestep = th.zeros((args.batch_size, )).to(dist_util.dev())
    
    classifier_data_argmax = dict()

    for row_id, image_row in enumerate(result_images):
        source_class = image_classes[row_id]
        row_oracle_scores = []
        
        for col_id, image_col in enumerate(image_row):
            target_class = target_classes[col_id]
            run_id = f"{source_class}_{target_class}"
            
            image_col = image_col.to(dist_util.dev())
            
            with th.no_grad():
                original_logits = classifier.call(image_col, timestep, preprocessing=False)
                original_pred = th.argmax(original_logits, dim=1)
                
            current_single_metrics_table_row = single_metrics_table[row_id*len(target_classes) + col_id]


            classifier_results = []
            for oracle in all_classifiers:     
                # get the prediction of the classifier
                with th.no_grad():
                    logits = oracle.call(image_col, timestep, preprocessing=False)
                    pred = th.argmax(logits, dim=1)
                
                if classifier_data_argmax.get(oracle.name) is None:
                    classifier_data_argmax[oracle.name] = np.zeros((len(image_classes), len(target_classes)))

                classifier_oracle_score = th.mean(th.eq(pred, original_pred) * 1.0).item()
                classifier_data_argmax[oracle.name][row_id][col_id] = classifier_oracle_score

                classifier_results.append(classifier_oracle_score)
                current_single_metrics_table_row[f"oracle_{oracle.name}"] = classifier_oracle_score
                
                try:
                    classifier_oracle_score_tcv = th.mean((pred == target_class).float()).item()
                    current_single_metrics_table_row[f"oracle_tcv_{oracle.name}"] = classifier_oracle_score_tcv
                except:
                    logger.log("Couldn't save metric, error occured")

                
            oracle_score  = (1.0*sum(classifier_results)) / len(classifier_results)
            row_oracle_scores.append(oracle_score)  
            
            
            # ensure directory and save raw oracle scores
            os.makedirs(f"{wandb.run.dir}/oracle", exist_ok=True)
            dfo = pd.DataFrame([classifier_results], columns=[c.name for c in all_classifiers])
            dfo.to_csv(f"{wandb.run.dir}/oracle/{run_id}.csv")

            
        oracle_scores.append(row_oracle_scores)
        

    # log the oracle scores
    df = pd.DataFrame(oracle_scores, columns=get_imagenet_class_name(args.data_path, target_classes), index=get_imagenet_class_name(args.data_path, image_classes))
    df.index.name = 'source_class'
    df = df.reset_index()
    table = wandb.Table(dataframe=df)
    wandb.log({"metric_oracle_scores": table})
    df.to_csv(f"{wandb.run.dir}/oracle_scores.csv")

    # log the classifier data
    for classifier in all_classifiers:
        df = pd.DataFrame(classifier_data_argmax[classifier.name], columns=get_imagenet_class_name(args.data_path, target_classes), index=get_imagenet_class_name(args.data_path, image_classes))
        df.index.name = 'source_class'
        df = df.reset_index()
        table = wandb.Table(dataframe=df)
        wandb.log({f"metric_oarg_{classifier.name}": table})
        df.to_csv(f"{wandb.run.dir}/classifier_{classifier.name}_argmax.csv")


    # store single_metrics_table in csv
    logger.log("Save the single_metrics_table")
    df = pd.DataFrame.from_records(single_metrics_table)
    df.to_csv(f"{wandb.run.dir}/single_metrics_table.csv", index=False)

    logger.log("Store the single_metrics_table in wandb")
    table = wandb.Table(dataframe=df)
    wandb.log({"single_metrics_table": table})
    

    ################################################################
    # /ORACLE SCORES
    ################################################################
            
    # Create a 6x11 wandb.Table of images
    # 1 row for each source class
    # 1 column for each target class
    # each cell contains the image of the source class, the image of the target class and the adversarial image
    # the first row contains the target class names

    logger.log("Store images in wandb")
    # create the table
    targets = ["source_class"] + get_imagenet_class_name(args.data_path, target_classes)
    # use row_result_image_grids
    data = []
    for row_id, image_row in enumerate(result_image_grids):
        source_class = image_classes[row_id]
        source_class_name = get_imagenet_class_name(args.data_path, [source_class])[0]
        row = [source_class_name]
        for col_id, image_col in enumerate(image_row):
            target_class = target_classes[col_id]
            target_class_name = get_imagenet_class_name(args.data_path, [target_class])[0]
            run_id = f"{source_class_name}({source_class})_{target_class_name}({target_class})".replace(" ", "-")
            row.append(image_col)
        data.append(row)
    table = wandb.Table(columns=targets, data=data)
    wandb.log({"images": table})


                
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        seed=1,
        num_samples=50,
        batch_size=50,
        use_ddim=False,
        model_path="",
        classifier_class="",
        classifier_path="",
        classifier_perturbation=0.0,
        classifier_scale=0.1,
        lp_custom=1.0,
        lp_regularization_scale=0.15,
        skip_timesteps=0,
        init_image=False,
        data_path="",
        cone_projection=False,
        xzero_prediction=True
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
