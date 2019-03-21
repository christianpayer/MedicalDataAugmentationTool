import itk
from glob import glob
import os


def reorient_to_reference(image, reference):
    filter = itk.OrientImageFilter[type(image), type(image)].New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    filter.SetDesiredCoordinateDirection(reference.GetDirection())
    filter.Update()
    return filter.GetOutput()


def cast(image, reference):
    filter = itk.CastImageFilter[type(image), type(reference)].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()


def copy_information(image, reference):
    filter = itk.ChangeInformationImageFilter[type(image)].New()
    filter.SetInput(image)
    filter.SetReferenceImage(reference)
    filter.UseReferenceImageOn()
    filter.ChangeSpacingOn()
    filter.ChangeOriginOn()
    filter.ChangeDirectionOn()
    filter.Update()
    return filter.GetOutput()


def relabel(labels):
    labels_np = itk.GetArrayViewFromImage(labels)
    from_labels = [1, 2, 3, 4, 5, 6, 7]
    to_labels = [500, 600, 420, 550, 205, 820, 850]
    for from_label, to_label in zip(from_labels, to_labels):
        labels_np[labels_np == from_label] = to_label


if __name__ == '__main__':
    # TODO: set to True for CT and False for MR
    is_ct = False
    # TODO: change folder to where the predictions are saved
    input_folder = 'TODO'
    # TODO: change folder to where original files (e.g. mr_train_1001_image.nii.gz and mr_train_1001_label.nii.gz) from MMWHS challenge are saved
    reference_folder = 'TODO' if is_ct else 'TODO'

    output_folder = './reoriented/ct/' if is_ct else './reoriented/mr/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = glob(input_folder + '*.mha')
    for filename in sorted(filenames):
        if 'prediction' in filename:
            continue
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.mha')]
        print(basename_wo_ext)
        image = itk.imread(filename)
        reference = itk.imread(os.path.join(reference_folder, basename_wo_ext + '_label.nii.gz'))
        reoriented = cast(image, reference)
        reoriented = reorient_to_reference(reoriented, reference)
        reoriented = copy_information(reoriented, reference)
        relabel(reoriented)
        itk.imwrite(reoriented, os.path.join(output_folder, basename_wo_ext + '.nii.gz'))
