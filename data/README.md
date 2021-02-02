# List Dataset
The class ListDataset use the dataset name and task name to load the respectives images accordingly, modify this class to your use case.

For a simple loader, this version is implemented considering the following folder scheme:

```
/datasets_root_folder
|--- dataset_1
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt
     ...
     |--- fold_k_img_list.txt
|--- dataset_2
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt
     ...
     |--- fold_k_img_list.txt
...
|--- dataset_n
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt
     ...
     |--- fold_k_img_list.txt
```

## Adaptation to your data
In the file `list_dataset.py` you can/have to make changes to your data.

### Setup /datasets_root_folder param:
Change the variable `root` to the your specific root folder path.

### make_dataset function:
The make_dataset fucntion:
```
# Returns a list of pairs (img_path, mask_path)
    def make_dataset(self):
        ...
```
Generate the list of image files that comprise your dataset, if you have a different folder tree structure, modify this function
to work with your file arrangements.


### get_data function:

The get_data fucntion:
```
# Returns: img, mask, (path_to_img, img_filename)

    def get_data(self, index):
        img_path, msk_path = self.imgs[index]
        ...

```

Loads the image and labels/mask objects. It was implemented to work with 1D images (ie, grayscale). If you have a different data type you
will have to modify this function. Also note, that we only consider binary problems, so if your data is multiclass you will have to binarize your masks
in this function.