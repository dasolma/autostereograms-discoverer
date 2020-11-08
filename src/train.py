from __future__ import absolute_import
import argparse
import sys
from models import classify_overlap, classifly_with_overlaping_layer, classifly_with_stereoconv_layer
from utils import in_ipynb
from livelossplot import PlotLossesKeras

def main(argv):
    """Command line entry point
    :param argv: command parameters (without command name is expected)
    """

    parser = argparse.ArgumentParser(prog='train', description='train the differents models')
    parser.add_argument('experiment', nargs='?', default=None, help='name of the experiment: naive')
    parser.add_argument('--test_path', help='path to the images used as test dataset', action='append',
                        required=False)
    parser.add_argument('--output', help='file path where save the trained model', action='append',
                        required=False)

    args = parser.parse_args(argv)

    callbacks = [PlotLossesKeras()] if in_ipynb() else []
    models = {
        'naive': classify_overlap,
        'overlapping': classifly_with_overlaping_layer,
        'stereo': classifly_with_stereoconv_layer
    }

    if args.experiment in models.keys():
        model, train_gen, val_gen = models[args.experiment]()
        model.fit_generator(generator=train_gen, validation_data=val_gen,
                            epochs=200, verbose=1, callbacks=callbacks)
        if args.output:
            model.save(args.output)

    else:
        print('Experiment unknown. Valid experiments are: %s' % ','.join(models.keys()))

if __name__ == '__main__':
    main(sys.argv[1:])
