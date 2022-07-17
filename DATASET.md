## Dataset 

### ShapNet55/34

We propose two more challenging benchmarks ShapeNet-55 and ShapeNet-34 with more diverse incomplete point clouds that can better reflect the real-world scenarios to promote future research. Our dataset is based on [ShapeNetCore](https://shapenet.org/). Compared to existing datasets like PCN, ShapeNet-55 considers more diverse tasks (i.e., upsampling and completion of point cloud), more diverse categories (i.e., from 8 categories to 55 categories), more diverse viewpoints (i.e., from 8 viewpoints to all possible viewpoints) and more diverse levels of incompleteness (i.e., missing 25% to 75% points of the groundtruth point clouds). We also propose to benchmark the completion perfomance on objects from unseen categories with ShapeNet-34.  

![dataset](fig/dataset.png)


### Data Preparation
The overall directory structure should be:

```
│PoinTr/
├──cfgs/
├──datasets/
├──data/
│   ├──ShapeNet55-34/
│   ├──PCN/
│   ├──KITTI/
├──.......
```
**ShapeNet55/34 Dataset**: You can download the processed ShapeNet55/34 dataset at [[BaiduCloud](https://pan.baidu.com/s/16Q-GsEXEHkXRhmcSZTY86A)] (code:le04) or [[Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)]. Unzip the file under `ShapeNet55-34/`. The directory structure should be

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-xxxxxxxxxxxxxx.npy
│  ├── 02691156-xxxxxxxxxxxxxx.npy
│  ├── .......
├──ShapeNet-34/
│  ├── train.txt
│  └── test.txt
├──ShapeNet-34/
│  ├── train.txt
│  └── test.txt
├──ShapeNet-Unseen21/
   └── test.txt
```

**PCN Dataset**: You can download the processed PCN dataset from this [url](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion). The directory structure should be

```
│PCN/
├──train/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx
│  │   │   │   ├── 00.pcd
│  │   │   │   ├── 01.pcd
│  │   │   │   ├── .......
│  │   │   │   └── 07.pcd
│  │   │   ├── .......
│  │   ├── .......
├──test/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──val/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── xxxxxxxxxxxxxx
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──PCN.json
└──category.txt
```

**KITTI**: You can download the KITTI dataset from this [url](https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F). The directory structure should be

```
│KITTI/
├──bboxes/
│  ├── frame_0_car_0.txt
│  ├── .......
├──cars/
│  ├── frame_0_car_0.pcd
│  ├── .......
├──tracklets/
│  ├── tracklet_0.txt
│  ├── .......
├──KITTI.json
```

##

##

##

##

##

##
