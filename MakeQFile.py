import os
from tqdm import tqdm

#path to all the  pictures
filepathlist = ['/home/mcamp/PycharmProjects/GarageDoor/ResizedImages/open/',
                '/home/mcamp/PycharmProjects/GarageDoor/ResizedImages/closed/']

#put all the paths+filenames into a list
def q_file_maker(filepath):
    filenames = []
    for path in filepathlist:
        image_files = os.listdir(path)
        for image in tqdm(image_files):
            image_file = os.path.join(path, image)
            image_file = image_file + ',' + os.path.basename(os.path.normpath(path))
            # image_file = image_file
            filenames.append(image_file)

    #write
    new_writefile = open("queue.txt", "w")
    for k in filenames:
        new_writefile.write("%s\n" % k)
    new_writefile.close()

q_file_maker(filepathlist)
