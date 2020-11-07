import argparse
import sys

from src.models import classify_overlap


def main(argv):
    """Command line entry point
    :param argv: command parameters (without command name is expected)
    """

    parser = argparse.ArgumentParser(prog='train', description='train the differents models')
    parser.add_argument('experiment', nargs='?', default=None, help='name of the experiment: naive')
    parser.add_argument('--data_path', help='path to the dataset', action='append',
                        required=False)
    parser.add_argument('--output', help='file path where save the trained model', action='append',
                        required=False)

    args = parser.parse_args(argv)

    if args.experiment == 'naive':
        model, train_gen, val_gen = classify_overlap()
        model.fit_generator(generator=train_gen,
                                validation_data=val_gen,
                                epochs=200, verbose=1)
        if args.output:
            model.save(args.output)

    else:
        print('Experiment unknown. Valid experiments are: naive')



if __name__ == '__main__':
    main(sys.argv[1:])
