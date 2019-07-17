import scipy.io as sio
import h5py

def load_deep_features(data_name):
    import numpy as np
    valid_data, req_rec, b_wv_matrix = True, True, True
    unlabels, zero_shot, doc2vec, split = False, False, False, False
    if data_name.find('_doc2vec') > -1:
        doc2vec = True
        req_rec, b_wv_matrix = False, False
    if data_name == 'wiki_doc2vec':
        path = './datasets/wiki_data/wiki_deep_doc2vec_data_corr_ae.h5py' # wiki_deep_doc2vec_data
        valid_len = 231
        MAP = -1
    elif data_name == 'nus_wide_doc2vec':
        path = './datasets/NUS-WIDE/nus_wide_deep_doc2vec_data_42941.h5py' # pascal_sentence_deep_doc2vec_data
        valid_len = 5000
        MAP = -1
    elif data_name == 'MSCOCO_doc2vec':
        path = './datasets/MSCOCO/MSCOCO_deep_doc2vec_data.h5py' #
        valid_len = 10000
        MAP = -1
    elif data_name == 'xmedia':
        path = './datasets/XMedia&Code/XMediaFeatures.mat'
        MAP = -1
        req_rec, b_wv_matrix = False, False
        all_data = sio.loadmat(path)
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature
        I_te_CNN = all_data['I_te_CNN'].astype('float32')	# Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')	# Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')	# Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')	# Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')	# Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')	# Features of training set for video(frame) data, CNN feature
        te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64')   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64')   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64') # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64') # category label of training set for audio data
        teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64') # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64') # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64') # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1]).astype('int64') # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64') # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64') # category label of training set for text data

        train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
        test_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], A_te[0: 100], d3_te[0: 50], V_te_CNN[0: 87]]
        valid_data = [I_te_CNN[500::], T_te_BOW[500::], A_te[100::], d3_te[50::], V_te_CNN[87::]]
        train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
        test_labels = [teImgCat[0: 500], teTxtCat[0: 500], teAudCat[0: 100], te3dCat[0: 50], teVidCat[0: 87]]
        valid_labels = [teImgCat[500::], teTxtCat[500::], teAudCat[100::], te3dCat[50::], teVidCat[87::]]


    if doc2vec:
        h = h5py.File(path)
        train_imgs_deep = h['train_imgs_deep'][()].astype('float32')
        train_imgs_labels = h['train_imgs_labels'][()]
        train_imgs_labels -= np.min(train_imgs_labels)
        train_texts_idx = h['train_text'][()].astype('float32')
        train_texts_labels = h['train_texts_labels'][()]
        train_texts_labels -= np.min(train_texts_labels)
        train_data = [train_imgs_deep, train_texts_idx]
        train_labels = [train_imgs_labels, train_texts_labels]

        # valid_data = False

        test_imgs_deep = h['test_imgs_deep'][()].astype('float32')
        test_imgs_labels = h['test_imgs_labels'][()]
        test_imgs_labels -= np.min(test_imgs_labels)
        test_texts_idx = h['test_text'][()].astype('float32')
        test_texts_labels = h['test_texts_labels'][()]
        test_texts_labels -= np.min(test_texts_labels)
        test_data = [test_imgs_deep, test_texts_idx]
        test_labels = [test_imgs_labels, test_texts_labels]

        valid_data = [test_data[0][0: valid_len], test_data[1][0: valid_len]]
        valid_labels = [test_labels[0][0: valid_len], test_labels[1][0: valid_len]]

        test_data = [test_data[0][valid_len::], test_data[1][valid_len::]]
        test_labels = [test_labels[0][valid_len::], test_labels[1][valid_len::]]

    if valid_data:
        if b_wv_matrix:
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, wv_matrix, MAP
        else:
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, MAP
    else:
        if b_wv_matrix:
            return train_data, train_labels, test_data, test_labels, wv_matrix, MAP
        else:
            return train_data, train_labels, test_data, test_labels, MAP
