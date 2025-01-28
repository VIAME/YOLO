#!/bin/bash
#export CUDA_VISIBLE_DEVICES="1,"
BUNDLE_DPATH=$HOME/data/dvc-repos/viame_dvc/private/Benthic/HABCAM-FISH
TRAIN_FPATH=$BUNDLE_DPATH/train-v05-noscallop.kwcoco.zip
VALI_FPATH=$BUNDLE_DPATH/vali-v05-noscallop.kwcoco.zip
TEST_FPATH=$BUNDLE_DPATH/test-v05-noscallop.kwcoco.zip



# In the current version we need to write configs to the repo itself.
# Its a bit gross, but this should be somewhat robust.
# Find where the yolo repo is installed (we need to be careful that this is the
# our fork of the WongKinYiu variant
MODULE_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent)")
CONFIG_DPATH=$(python -c "import yolo.config, pathlib; print(pathlib.Path(yolo.config.__file__).parent / 'dataset')")
echo "REPO_DPATH = $REPO_DPATH"
echo "MODULE_DPATH = $MODULE_DPATH"
echo "CONFIG_DPATH = $CONFIG_DPATH"
DATASET_CONFIG_FPATH=$CONFIG_DPATH/viame-v05-noscallop.yaml

# Hack to construct the class part of the YAML
CLASS_YAML=$(python -c "if 1:
    import kwcoco
    train_fpath = kwcoco.CocoDataset('$TRAIN_FPATH')
    categories = train_fpath.categories().objs
    # It would be nice to have better class introspection, but in the meantime
    # do the same sorting as yolo.tools.data_conversion.discretize_categories
    categories = sorted(categories, key=lambda cat: cat['id'])
    class_num = len(categories)
    class_list = [c['name'] for c in categories]
    print(f'class_num: {class_num}')
    print(f'class_list: {class_list}')
")


CONFIG_YAML="
path: $BUNDLE_DPATH
train: $TRAIN_FPATH
validation: $VALI_FPATH

$CLASS_YAML
"
echo "$CONFIG_YAML" > "$DATASET_CONFIG_FPATH"


# This might only work in development mode, otherwise we will get site packages
# That still might be fine, but we do want to fix this to run anywhere.
REPO_DPATH=$(python -c "import yolo, pathlib; print(pathlib.Path(yolo.__file__).parent.parent)")
cd "$REPO_DPATH"
#export CUDA_LAUNCH_BLOCKING=1
#DISABLE_RICH_HANDLER=1
LOG_BATCH_VIZ_TO_DISK=1 python yolo/lazy.py \
    task=train \
    dataset=viame-v05-noscallop \
    use_wandb=False \
    out_path=viame-runs \
    name=viame-test \
    cpu_num=8 \
    accelerator=auto \
    task.data.batch_size=4 \
    task.optimizer.args.lr=0.003

#device=0 \
#"image_size=[224,224]" \
#--help

# Grab a checkpoint
CKPT_FPATH=$(python -c "if 1:
    import pathlib
    ckpt_dpath = pathlib.Path('.') / 'viame-runs/train/viame-test/checkpoints'
    checkpoints = sorted(ckpt_dpath.glob('*'))
    print(checkpoints[-1])
")
echo "CKPT_FPATH=$CKPT_FPATH"

# Get a single image to test on
FNAME=$(kwcoco info "$TEST_FPATH" --show_images 10 | jq -r ".images[2] | .file_name")
FPATH=$BUNDLE_DPATH/$FNAME
echo "FPATH = $FPATH"

export CUDA_VISIBLE_DEVICES="1,"
python yolo/lazy.py \
    task.data.source="$TEST_FPATH" \
    task=inference \
    dataset=viame-v05-noscallop \
    use_wandb=False \
    out_path=viame-inference \
    name=viame-inference-test \
    cpu_num=8 \
    weight="\"$CKPT_FPATH\"" \
    accelerator=auto
