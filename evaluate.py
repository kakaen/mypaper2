from math import exp
from pathlib import Path

import numpy as np
import rasterio


def run_once_evaluate(_root_dir: Path, evalPerBand=False):
    save_name = _root_dir.name

    labels = {
        "MAE": False,
        "RMSE": True,
        "SAM": True,
        "SSIM": True,
        "ERGAS": True,
        "PSNR": True,
        "UIQI": True,
        "CC": True
    }

    # ---------------------------------

    # --------------图像参数------------
    MAX_VALUE = 65535  # 16位,缩放前
    scaler_factor = 0.0001

    # ----------------------------------

    MAX_VALUE *= scaler_factor

    if evalPerBand is True:
        labels["SAM"] = False

    def read_tif(path):
        with rasterio.open(str(path)) as ds:
            im = ds.read().astype(np.float32)[:, 0:, 0:]
        return im

    def MAE(_img1, _img2):
        mae = np.mean(np.abs(_img1 - _img2))
        return mae

    def RMSE(_img1, _img2):
        mse = np.mean((_img1 - _img2) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def SAM(y_true, y_pred):
        """Spectral Angle Mapper"""
        y_true = y_true.astype(np.float32)
        y_true = y_true + 1 ** (-5)
        y_pred = y_pred.astype(np.float32)
        y_pred = y_pred + 1 ** (-5)
        y_true_prod = np.sum(np.sqrt(y_true ** 2), axis=0)
        y_pred_prod = np.sum(np.sqrt(y_pred ** 2), axis=0)
        true_pred_prod = np.sum(y_true * y_pred, axis=0)
        ratio = true_pred_prod / (y_true_prod * y_pred_prod)
        angle = np.mean(np.arccos(ratio))
        return angle
    """新添加SAM计算函数"""
    def SAM_1(y_true, y_pred):
        
        """Spectral Angle Mapper fixed"""
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        y_true[y_true<0] = 0
        y_pred[y_pred<0] = 0
        y_true = y_true+10**(-5)  
        y_pred = y_pred+10**(-5)

        
        y_true_prod = np.sqrt(np.sum(y_true ** 2, axis=0))
        y_pred_prod = np.sqrt(np.sum(y_pred ** 2, axis=0))

        true_pred_prod = np.sum(y_true * y_pred, axis=0)
        ratio = true_pred_prod / (y_true_prod * y_pred_prod)
        ratio = np.clip(ratio,-1,1)
        angle = np.mean(np.arccos(ratio))
        return angle
    
    def ERGAS(y_true, y_pred, scale_factor=16):
        errors = []
        for _i in range(y_true.shape[0]):
            errors.append(RMSE(y_true[_i], y_pred[_i]))
            errors[_i] /= np.mean(y_pred[_i])
        return 100.0 / scale_factor * np.sqrt(np.mean(errors))

    def PSNR(_img1, _img2):
        mse = np.mean((_img1 - _img2) ** 2)
        if mse == 0:
            return float('inf')
        else:
            return 20 * np.log10(MAX_VALUE / np.sqrt(mse))

    from skimage.metrics import structural_similarity as ssim

    def SSIM(_img1, _img2):

        return ssim(_img1.transpose(1, 2, 0), _img2.transpose(1, 2, 0), multichannel=True, data_range=10000)

    def _UIQI(im1, im2, block_size=64, return_map=False):
        if len(im1.shape) == 3:
            return np.array(
                [UIQI(im1[:, :, _i], im2[:, :, _i], block_size, return_map=return_map) for _i in range(im1.shape[2])])
        delta_x = np.std(im1, ddof=1)
        delta_y = np.std(im2, ddof=1)
        delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))
                          ) / (im1.shape[0] * im1.shape[1] - 1)
        mu_x = np.mean(im1)
        mu_y = np.mean(im2)
        q1 = delta_xy / (delta_x * delta_y)
        q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
        q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
        q = q1 * q2 * q3
        return q

    def UIQI(im1, im2, block_size=64, return_map=False):
        cur = []
        for _i in range(im1.shape[0]):
            cur.append(_UIQI(im1[_i], im2[_i]))
        return np.mean(np.array(cur))

    def CC(_img1, _img2):
        cur_cc = np.sum((_img1 - np.mean(_img1)) * (_img2 - np.mean(_img2))) / \
                 np.sqrt((np.sum(np.square(_img1 - np.mean(_img1)))) *
                         np.sum(np.square(_img2 - np.mean(_img2))))
        return cur_cc

    def get_imgs_from_dir(_dir: Path):
        imgs = []
        for path in _dir.glob("*.tif"):
            img = read_tif(path)
            imgs.append(img)
        assert len(imgs) == 2
        return imgs[0], imgs[1], _dir.name

    def calculate(_img1, _img2):

        val_dic = {}
        for key in labels.keys():
            val_dic[key] = None

        if labels["MAE"]:
            val_dic["MAE"] = float(MAE(_img1 * scaler_factor, _img2 * scaler_factor))
        if labels["RMSE"]:
            val_dic["RMSE"] = float(
                RMSE(_img1 * scaler_factor, _img2 * scaler_factor))
        if labels["SAM"]:
            val_dic["SAM"] = float(SAM_1(_img1, _img2))
        if labels["SSIM"]:
            val_dic["SSIM"] = float(SSIM(_img1, _img2))
        if labels["ERGAS"]:
            val_dic["ERGAS"] = float(
                ERGAS(_img1 * scaler_factor, _img2 * scaler_factor))
        if labels["PSNR"]:
            val_dic["PSNR"] = float(
                PSNR(_img1 * scaler_factor, _img2 * scaler_factor))
        if labels["UIQI"]:
            val_dic["UIQI"] = float(UIQI(_img1, _img2))
        if labels["CC"]:
            val_dic["CC"] = float(CC(_img1 * scaler_factor, _img2 * scaler_factor))
        return val_dic

    import csv

    def log_csv(filepath, values, header=None, multirows=False):
        empty = False
        if not filepath.exists():
            filepath.touch()
            empty = True

        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            if empty and header:
                writer.writerow(header)
            if multirows:
                writer.writerows(values)
            else:
                writer.writerow(values)

    # 执行评估
    dirs_list = [p for p in _root_dir.iterdir() if p.is_dir()]
    result_dic = {}

    for p in dirs_list:
        img1, img2, dir_name = get_imgs_from_dir(p)
        val_dic_p = {}
        if evalPerBand is not True:
            val_dic_p = calculate(img1, img2)
            result_dic[dir_name] = val_dic_p
        else:
            for i in range(img1.shape[0]):
                name = 'band' + str(i)
                val_dic_1band = calculate(img1[i:i + 1], img2[i:i + 1])
                val_dic_p[name] = val_dic_1band
            result_dic[dir_name] = val_dic_p
    if evalPerBand is not True:
        csv_path = _root_dir / (save_name + '.csv')
    else:
        csv_path = _root_dir / (save_name + 'perBand.csv')
    csv_header = ["time"]

    if evalPerBand is True:
        csv_header.append('band')

    for k, v in labels.items():
        if v:
            csv_header.append(k)

    if evalPerBand is not True:
        for k, v in result_dic.items():
            csv_value = [k]
            for kk, vv in labels.items():
                if vv:
                    csv_value.append(v[kk])
            log_csv(csv_path, csv_value, csv_header)

        #添加均值行
        mean_value=[]
        mean_value.append('mean')
        for kk,vv in labels.items():
            if vv:
                sum_vv=0
                #计算均值
                count=0
                for k,v in result_dic.items():
                    sum_vv+=v[kk]
                    count+=1
                mean_value.append(sum_vv/count)
        log_csv(csv_path, mean_value, csv_header)
    else:
        for k, v in result_dic.items():
            # csv_value = []
            # csv_value.append(k)     # model name
            dic_bands_vals = result_dic[k]
            for iband, valsDic in dic_bands_vals.items():
                csv_value = [k, iband]
                for kk, vv in labels.items():
                    if vv:
                        csv_value.append(valsDic[kk])
                log_csv(csv_path, csv_value, csv_header)
    
        #添加均值行
        mean_value=[]
        mean_value.append('mean')
        for kk,vv in labels.items():
            if vv:
                sum_vv=0
                #计算均值
                count=0
                for k,v in result_dic.items():
                    sum_vv+=v[kk]
                    count+=1
                mean_value.append(sum_vv/count)
        log_csv(csv_path, mean_value, csv_header)


def evaluate_results(saveimage_pairs: Path):
    run_once_evaluate(_root_dir=saveimage_pairs, evalPerBand=False)

# evaluate_results(Path("/home/u2023170762/save_dir/MLFFGAN对比/epoch=200/test"))