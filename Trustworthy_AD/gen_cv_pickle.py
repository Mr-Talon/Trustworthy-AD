from __future__ import print_function, division
import argparse
import pickle

import Trustworthy_AD.dataloader as dataloader
import Trustworthy_AD.misc as misc


def get_data(args, fields):
    ret = {}
    train_mask, pred_mask, pred_mask_frame = misc.get_mask(args.mask, args.validation)
    ret['baseline'], ret['pred_start'] = misc.get_baseline_prediction_start(pred_mask_frame)
    ret['duration'] = 7 * 12

    columns = ['RID', 'Month_bl', 'DX'] + fields
    frame = misc.load_table(args.spreadsheet, columns)

    tf = frame.loc[train_mask, fields]
    ret['mean'] = tf.mean()
    ret['stds'] = tf.std()
    ret['VentICVstd'] = (tf['Ventricles'] / tf['ICV']).std()

    frame[fields] = (frame[fields] - ret['mean']) / ret['stds']

    default_val = {f: 0. for f in fields}
    default_val['DX'] = 0.

    data = dataloader.extract(frame[train_mask], args.strategy, fields, default_val)
    ret['train'] = dataloader.Random(data, args.batch_size, fields)

    data = dataloader.extract(frame[pred_mask], args.strategy, fields, default_val)
    ret['test'] = dataloader.Sorted(data, 1, fields)

    print('train', len(ret['train'].subjects), 'subjects')
    print('test', len(ret['test'].subjects), 'subjects')
    print(len(fields), 'features')

    return ret


def main():
    for i in range(10):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mask',default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/Trustworthy_AD/Trustworthy_AD/folds/fold'+str(i)+'_mask.csv')
        parser.add_argument('--strategy', default='model')
        parser.add_argument('--spreadsheet', default='C:/Users/16046/Desktop/Programming/python'
                                                     '/深度学习/证据深度学习/Trustworthy_AD/data/TADPOLE_D1_D2.csv')
        parser.add_argument('--features', default='C:/Users/16046/Desktop/Programming/python'
                                                  '/深度学习/证据深度学习/Trustworthy_AD/data/features')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--out', default='C:/Users/16046/Desktop/Programming/python/深度学习/证据深度学习/Trustworthy_AD/Trustworthy_AD/traindata/train'+str(i)+'.pkl')
        parser.add_argument('--validation', action='store_true',default=True)
        args = parser.parse_args()

        print('1>> ', args.out)
        print('2>> ', args.mask)

        fields = misc.load_feature(args.features)
        with open(args.out, 'wb') as fhandler:
            pickle.dump(get_data(args, fields), fhandler)


if __name__ == '__main__':
    main()
