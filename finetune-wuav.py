import tensorflow as tf
from tensorflow import keras
import argparse
from m4depth_options import M4DepthOptions
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import dataloaders as dl
from callbacks import *
from m4depth_network_v2 import *
from metrics import *

if __name__ == '__main__':
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation
    try:
        # Manage GPU memory to be able to run the validation step in parallel on the same GPU
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()

    wuav_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                           os.path.join(*[cmd.records_path, "wilduav", "train_data"]),
                                           4, 4, True)
    wuav_dataloader = dl.get_loader("wilduav")
    wuav_dataloader.get_dataset("finetune", wuav_params, batch_size=cmd.batch_size)

    #midair_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                           #os.path.join(*[cmd.records_path, "midair", "train_data"]),
                                           #8, 4, True)
    #midair_dataloader = dl.get_loader("midair")
    #midair_dataloader.get_dataset("finetune", midair_params, batch_size=cmd.batch_size, out_size=airsim_dataloader.out_size, crop=True)

    #joint_dataset_length = airsim_dataloader.length*2
    #joint_dataset = tf.data.Dataset.sample_from_datasets([airsim_dataloader.dataset.repeat(), midair_dataloader.dataset.repeat()], weights=[0.5, 0.5], stop_on_empty_dataset=True)
    
    joint_dataset_length = wuav_dataloader.length
    joint_dataset = tf.data.Dataset.sample_from_datasets([wuav_dataloader.dataset.repeat()], weights=[1.0], stop_on_empty_dataset=True)
    print("LENGTH ", joint_dataset_length)
    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    # print("Training on %s" % cmd.dataset)
    tf.random.set_seed(42)
    data = joint_dataset

    model = M4Depth(nbre_levels=nbre_levels,
                    ablation_settings=model_opts.ablation_settings,
                    is_training=True, num_classes = wuav_dataloader.class_count)

    tensorboard_cbk = keras.callbacks.TensorBoard(
        log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
        write_images=False, update_freq=1200,
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
    # stop_nan_cbk = ks.callbacks.TerminateOnNaN()
    model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model.compile(optimizer=opt, metrics=[tf.keras.metrics.MeanIoU(num_classes = wuav_dataloader.class_count)])

    if enable_validation:
        val_cbk = [CustomKittiValidationCallback(cmd)]
    else:
        val_cbk = []
    print(joint_dataset_length)
    #model.fit(data, epochs=model_checkpoint_cbk.resume_epoch + (20000 // joint_dataset_length) + 1,
    model.fit(data, epochs=model_checkpoint_cbk.resume_epoch + (61 // joint_dataset_length) + 1,
                initial_epoch=model_checkpoint_cbk.resume_epoch,
                steps_per_epoch= joint_dataset_length,
                callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)
