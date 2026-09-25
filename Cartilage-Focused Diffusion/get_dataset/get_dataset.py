from torch.utils.data import DataLoader
from dataset import PFOADataset, PFOAInDataset
from dataset import PFOADataset_PDW_ATSE, PFOAInDataset_PDW_ATSE
from dataset import PFOADataset_T1W, PFOAInDataset_T1W


def get_inference_dataloader(dataset_root_dir, test_txt_dir,batch_size=1, drop_last=False, data_type=''):
    if data_type == 'pfoa':
        train_dataset = PFOAInDataset(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'pfoa_pdw_atse':
        train_dataset = PFOAInDataset_PDW_ATSE(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    elif data_type == 'pfoa_t1w':
        train_dataset = PFOAInDataset_T1W(root_dir=dataset_root_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )
    return loader

def get_train_dataset(cfg):
    if cfg.dataset.data_type == 'pfoa':
        train_dataset = PFOADataset(root_dir=cfg.dataset.root_dir)
        sampler = None
    elif cfg.dataset.data_type == 'pfoa_pdw_atse':
        train_dataset = PFOADataset_PDW_ATSE(root_dir=cfg.dataset.root_dir)
        sampler = None
    elif cfg.dataset.data_type == 'pfoa_t1w':
        train_dataset = PFOADataset_T1W(root_dir=cfg.dataset.root_dir)
        sampler = None
    return train_dataset, sampler


