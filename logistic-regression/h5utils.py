from random import shuffle
import glob
import numpy as np
import h5py
import cv2

def images_to_h5():
    shuffle_data = True  # shuffle the addresses before saving
    hdf5_path = 'dataset/dataset.hdf5'  # address to where you want to save the hdf5 file
    train_path_0 = 'dataset/train/digit_0/*.png'
    train_path_1 = 'dataset/train/digit_1/*.png'
    train_path_2 = 'dataset/train/digit_2/*.png'
    train_path_3 = 'dataset/train/digit_3/*.png'
    train_path_4 = 'dataset/train/digit_4/*.png'
    train_path_5 = 'dataset/train/digit_5/*.png'
    train_path_6 = 'dataset/train/digit_6/*.png'
    train_path_7 = 'dataset/train/digit_7/*.png'
    train_path_8 = 'dataset/train/digit_8/*.png'
    train_path_9 = 'dataset/train/digit_9/*.png'

    train_path_10 = 'dataset/train/character_1_ka/*.png'
    train_path_11 = 'dataset/train/character_2_kha/*.png'
    train_path_12 = 'dataset/train/character_3_ga/*.png'
    train_path_13 = 'dataset/train/character_4_gha/*.png'
    train_path_14 = 'dataset/train/character_5_kna/*.png'
    train_path_15 = 'dataset/train/character_6_cha/*.png'
    train_path_16 = 'dataset/train/character_7_chha/*.png'
    train_path_17 = 'dataset/train/character_8_ja/*.png'
    train_path_18 = 'dataset/train/character_9_jha/*.png'
    train_path_19 = 'dataset/train/character_10_yna/*.png'
    train_path_20 = 'dataset/train/character_11_taamatar/*.png'
    train_path_21 = 'dataset/train/character_12_thaa/*.png'
    train_path_22 = 'dataset/train/character_13_daa/*.png'
    train_path_23 = 'dataset/train/character_14_dhaa/*.png'
    train_path_24 = 'dataset/train/character_15_adna/*.png'
    train_path_25 = 'dataset/train/character_16_tabala/*.png'
    train_path_26 = 'dataset/train/character_17_tha/*.png'
    train_path_27 = 'dataset/train/character_18_da/*.png'
    train_path_28 = 'dataset/train/character_19_dha/*.png'
    train_path_29 = 'dataset/train/character_20_na/*.png'
    train_path_30 = 'dataset/train/character_21_pa/*.png'
    train_path_31 = 'dataset/train/character_22_pha/*.png'
    train_path_32 = 'dataset/train/character_23_ba/*.png'
    train_path_33 = 'dataset/train/character_24_bha/*.png'
    train_path_34 = 'dataset/train/character_25_ma/*.png'
    train_path_35 = 'dataset/train/character_26_yaw/*.png'
    train_path_36 = 'dataset/train/character_27_ra/*.png'
    train_path_37 = 'dataset/train/character_28_la/*.png'
    train_path_38 = 'dataset/train/character_29_waw/*.png'
    train_path_39 = 'dataset/train/character_30_motosaw/*.png'
    train_path_40 = 'dataset/train/character_31_petchiryakha/*.png'
    train_path_41 = 'dataset/train/character_32_patalosaw/*.png'
    train_path_42 = 'dataset/train/character_33_ha/*.png'
    train_path_43 = 'dataset/train/character_34_chhya/*.png'
    train_path_44 = 'dataset/train/character_35_tra/*.png'
    train_path_45 = 'dataset/train/character_36_gya/*.png'

    train_paths = [train_path_0,train_path_1,train_path_2,train_path_3,train_path_4,train_path_5,train_path_6,train_path_7,train_path_8,train_path_9,
    train_path_10,train_path_11,train_path_12,train_path_13,train_path_14,train_path_15,train_path_16,train_path_17,train_path_18,train_path_19,
    train_path_20,train_path_21,train_path_22,train_path_23,train_path_24,train_path_25,train_path_26,train_path_27,train_path_28,train_path_29,
    train_path_30,train_path_31,train_path_32,train_path_33,train_path_34,train_path_35,train_path_36,train_path_37,train_path_38,train_path_39,
    train_path_40,train_path_41,train_path_42,train_path_43,train_path_44,train_path_45]

    # read addresses and labels from the 'train' folder
    addrs = []
    for i in range(46):
        print(train_paths[i])
        address = glob.glob(train_paths[i])
        addrs.extend(address)


    def getLabels(addr):
        label = -1
        if 'digit_0' in addr:
            label = 0
        elif 'digit_1' in addr:
            label = 1
        elif 'digit_2' in addr:
            label = 2
        elif 'digit_3' in addr:
            label = 3
        elif 'digit_4' in addr:
            label = 4
        elif 'digit_5' in addr:
            label = 5
        elif 'digit_6' in addr:
            label = 6
        elif 'digit_7' in addr:
            label = 7
        elif 'digit_8' in addr:
            label = 8
        elif 'digit_9' in addr:
            label = 9

        elif 'character_1_ka' in addr:
            label = 10
        elif 'character_2_kha' in addr:
            label = 11
        elif 'character_3_ga' in addr:
            label = 12
        elif 'character_4_gha' in addr:
            label = 13
        elif 'character_5_kna' in addr:
            label = 14
        elif 'character_6_cha' in addr:
            label = 15
        elif 'character_7_chha' in addr:
            label = 16
        elif 'character_8_ja' in addr:
            label = 17
        elif 'character_9_jha' in addr:
            label = 18
        elif 'character_10_yna' in addr:
            label = 19

        elif 'character_11_taamatar' in addr:
            label = 20
        elif 'character_12_thaa' in addr:
            label = 21
        elif 'character_13_daa' in addr:
            label = 22
        elif 'character_14_dhaa' in addr:
            label = 23
        elif 'character_15_adna' in addr:
            label = 24
        elif 'character_16_tabala' in addr:
            label = 25
        elif 'character_17_tha' in addr:
            label = 26
        elif 'character_18_da' in addr:
            label = 27
        elif 'character_19_dha' in addr:
            label = 28
        elif 'character_20_na' in addr:
            label = 29

        elif 'character_21_pa' in addr:
            label = 30
        elif 'character_22_pha' in addr:
            label = 31
        elif 'character_23_ba' in addr:
            label = 32
        elif 'character_24_bha' in addr:
            label = 33
        elif 'character_25_ma' in addr:
            label = 34
        elif 'character_26_yaw' in addr:
            label = 35
        elif 'character_27_ra' in addr:
            label = 36
        elif 'character_28_la' in addr:
            label = 37
        elif 'character_29_waw' in addr:
            label = 38
        elif 'character_30_motosaw' in addr:
            label = 39

        elif 'character_31_petchiryakha' in addr:
            label = 40
        elif 'character_32_patalosaw' in addr:
            label = 41
        elif 'character_33_ha' in addr:
            label = 42
        elif 'character_34_chhya' in addr:
            label = 43
        elif 'character_35_tra' in addr:
            label = 44
        elif 'character_36_gya' in addr:
            label = 45
        return label

    labels = [getLabels(addr) for addr in addrs]
    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]

    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]

    train_shape = (len(train_addrs), 32, 32)
    val_shape = (len(val_addrs), 32, 32)
    test_shape = (len(test_addrs), 32, 32)

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(hdf5_path, mode='w')

    hdf5_file.create_dataset("train_img", train_shape, np.int8)
    hdf5_file.create_dataset("val_img", val_shape, np.int8)
    hdf5_file.create_dataset("test_img", test_shape, np.int8)

    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
    hdf5_file["train_labels"][...] = train_labels
    hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
    hdf5_file["val_labels"][...] = val_labels
    hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
    hdf5_file["test_labels"][...] = test_labels

    mean = np.zeros(train_shape[1:], np.float32)

    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print("Train data: {}/{}".format(i, len(train_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add any image pre-processing here

        # save the image and calculate the mean so far
        hdf5_file["train_img"][i, ...] = img[None]
        mean += img / float(len(train_labels))

    # loop over validation addresses
    for i in range(len(val_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print("Validation data: {}/{}".format(i, len(val_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = val_addrs[i]
        img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add any image pre-processing here

        # save the image
        hdf5_file["val_img"][i, ...] = img[None]

        # loop over test addresses
    for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print("Test data: {}/{}".format(i, len(test_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = test_addrs[i]
        img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # add any image pre-processing here
        # save the image
        hdf5_file["test_img"][i, ...] = img[None]

    # save the mean and close the hdf5 file
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
