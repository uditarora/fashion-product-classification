import argparse
import logging
import os
from src.train import setup_top20, setup_ft, setup_bottom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fashion')

def main(data_path, ckpt_path):
    # Train on top-20 classes (train subsplit)
    logger.info("Training on top-20 classes")
    ckpt_path_top20 = os.path.join(ckpt_path, 'best_val_top20.ckpt')
    processor, trainer_top20, _ = setup_top20(ckpt_path=ckpt_path_top20,
        data_path=data_path)
    trainer_top20.train(25)

    acc_df = trainer_top20.get_test_accuracy()
    print("Test accuracy for top-20 classes:")
    print(acc_df)

    # Train on remaining classes (fine-tune subsplit)
    logger.info("Training on fine-tune subsplit")
    ckpt_path_ft = os.path.join(ckpt_path, 'best_val_ft.ckpt')
    processor, trainer_ft, _ = setup_ft(processor=processor,
        ckpt_path=ckpt_path_ft, model=trainer_top20.get_best_model())
    trainer_ft.train(50)

    acc_df_ft = trainer_ft.get_test_accuracy()
    print("Test accuracy for fine-tune classes:")
    print(acc_df_ft)

    # Train on bottom 50 classes of fine-tune subsplit
    # with alternate data augmentations
    logger.info("Training on bottom 50 classes of fine-tune subsplit")
    trainer_b50 = setup_bottom(processor, trainer_ft, num=50)
    trainer_b50.train(20)

    acc_df_ft2 = trainer_ft.get_test_accuracy()
    print("Test accuracy for fine-tune data after second round of training:")
    print(acc_df_ft2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="path to the dataset", required=True)
    parser.add_argument("--ckpt", help="path to checkpoint folder", default='ckpts')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)

    main(args.data, args.ckpt)
