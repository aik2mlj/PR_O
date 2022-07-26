from polydis_origin_dataset.dataset_loaders_pod import MusicDataLoaders
from config import prepare_model, train_model, FINETUNE_CONFIG, result_path_folder_path
from polydis_origin_dataset.dataset_pod import SEED

batch_size = 128
run_epochs = 40

if __name__ == "__main__":
    model_id = "finetune_txtenc"
    dataset_id = "polydis_origin"
    model = prepare_model(model_id)
    result_path = result_path_folder_path(model_id + "+" + dataset_id)

    data_loaders = MusicDataLoaders.get_loaders(
        SEED,
        bs_train=batch_size,
        bs_val=batch_size,
        portion=8,
        shift_low=-6,
        shift_high=5,
        num_bar=2,
        contain_chord=True,
    )

    train_model(
        model=model,
        data_loaders=data_loaders,
        readme_fn=__file__,
        n_epoch=FINETUNE_CONFIG['n_epoch'],
        parallel=FINETUNE_CONFIG['parallel'],
        lr=FINETUNE_CONFIG['lr'],
        writer_names=model.writer_names,
        load_data_at_start=FINETUNE_CONFIG['load_data_at_start'],
        beta=FINETUNE_CONFIG['beta'],
        run_epochs=run_epochs,
        result_path=result_path
    )
