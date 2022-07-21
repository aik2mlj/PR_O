from config import TrainingCall
from datetime import datetime

# If test_mode is True, will load a mini dataset to debug the code.
test_mode = False

# model_id in ['prvae', 'prvae_pttxtenc', 'prvae_pttxtenc_contra', 'prvae_contra',
#        'finetune_txtenc']
model_id = "finetune_txtenc"

# A pre-trained model path. Probably never used.
model_path = None

# In case the training is terminated, checkpoint saving is not implemented in
# amc_dl. Manually enter the number of epochs trained so far. Default None.
run_epochs = None

if __name__ == '__main__':
    print(
        f"Start training: {model_id}, test_mode={test_mode} at {datetime.now().strftime('%m-%d_%H:%M:%S')}"
    )
    training = TrainingCall(model_id=model_id)
    training(
        test_mode=test_mode,
        model_path=model_path,
        run_epochs=run_epochs,
        readme_fn=__file__
    )
