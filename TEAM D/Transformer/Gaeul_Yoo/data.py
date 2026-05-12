from torch.utils.data import Dataset
import util


class fr_to_en(Dataset):
   # pytorch dataloader 사용을 위한 class

    def __init__(self, set_type):
        super().__init__()
        if set_type == "train":
            self.src_lang = util.open_text_set("data/train/train.fr")
            self.trg_lang = util.open_text_set("data/train/train.en")

            print('► Dataset is "train"')

        elif set_type == "valid":
            self.src_lang = util.open_text_set("data/valid/val.fr")
            self.trg_lang = util.open_text_set("data/valid/val.en")

            print('► Dataset is "valid"')

        else:
            raise ValueError('set_type must be "train" or "valid"')

    def __len__(self):
        return len(self.src_lang)

    def __getitem__(self, idx):
        return self.src_lang[idx], self.trg_lang[idx]