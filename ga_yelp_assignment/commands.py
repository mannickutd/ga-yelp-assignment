from flask.ext.script import Command, Manager, Option


class LabelCounts(Command):

    option_list = (
        Option('--out', '-o', dest='out_file'),
    )

    def run(self, out_file):
        from ga_yelp_assignment.utils import get_label_counts
        label_dict = get_label_counts(application.config.get('TRAIN_LABELS_CSV'))
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label', 'count'])
            writer.writerows(label_dict.items())


class PhotoSizes(Command):

    option_list = (
        Option('--out', '-o', dest='out_file'),
    )

    def run(self, out_file):
        from ga_yelp_assignment.utils import get_set_of_image_sizes
        img_size_dict = get_set_of_image_sizes(
            application.config.get('PHOTOS_DIRECTORY'))
        with open(out_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['size', 'count'])
            writer.writerows(img_size_dict.items())
