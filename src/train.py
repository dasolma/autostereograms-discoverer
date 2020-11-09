from __future__ import absolute_import
import argparse
import sys
from models import classify_overlap, classifly_with_overlaping_layer, classifly_with_stereoconv_layer
from utils import in_ipynb
from livelossplot import PlotLossesKerasTF
import os

def main(argv):
    """Command line entry point
    :param argv: command parameters (without command name is expected)
    """

    parser = argparse.ArgumentParser(prog='train', description='train the differents models')
    parser.add_argument('experiment', nargs='?', default=None, help='name of the experiment: naive')
    parser.add_argument('--test_path', help='path to the images used as test dataset', action='store',
                        required=False)
    parser.add_argument('--output', help='file path where save the trained model', action='store',
                        required=False)
    parser.add_argument('--gpu', help='the GPU id to use', action='store',
                        required=False)
    parser.add_argument('--epochs', help='number to epochs for training (default 200', action='store',
                        required=False, default=200, type=int)

    args = parser.parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    callbacks = [PlotLossesKerasTF()] if in_ipynb() else []
    models = {
        'naive': classify_overlap,
        'overlapping': classifly_with_overlaping_layer,
        'stereo': classifly_with_stereoconv_layer
    }

    if args.experiment in models.keys():
        model, train_gen, val_gen = models[args.experiment]()
        model.summary()

        print("Training %d epochs" % args.epochs)
        model.fit_generator(generator=train_gen, validation_data=val_gen,
                            epochs=args.epochs, verbose=1, callbacks=callbacks)
        if args.output:
            print("Saving model to ", args.output)
            model.save(args.output)

    else:
        print('Experiment unknown. Valid experiments are: %s' % ','.join(models.keys()))

if __name__ == '__main__':
    main(sys.argv[1:])
