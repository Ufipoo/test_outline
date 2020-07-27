import time
import os
if __name__ == '__main__':

    def enum(file):
        c = file.readlines()
        for line in c:
            schet = line.split()
            return int(schet[-1])
    def len_match(file):
        link_match = []
        spisok = file.readlines()
        for line in spisok:
            parts = line.split()
            for i in range(0, len(parts)):
                link_match.append(parts[i])
        return len(link_match)


    file_path1 = 'dataset_parsing.py'
    file_path2 = 'validation_hero.py'
    file_path3 = 'players_parsing.py'

    """while enum(open('D:/model_param/new_dataset_info/schet_valid.txt', 'r')) != 2480:
        time.sleep(1)
        os.system('python %s' % file_path2)
        print('++++')
        #print(enum(open('number_match.txt', 'r')))
    while enum(open('D:/model_param/new_dataset_info/schet_dop_data.txt', 'r')) != 2000:
        time.sleep(3)
        os.system('python %s' % file_path1)
        print('++++')"""
    while len_match(open('D:/model_param/new_dataset_info/valid_links.txt', 'r')) != 2001:
        time.sleep(1)
        os.system('python %s' % file_path3)

