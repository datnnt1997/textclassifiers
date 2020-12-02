import sys
import os

if __name__ == '__main__':
	filepath = "./processed_dev.txt"
	path, filename = os.path.split(filepath)
	name, ext = os.path.splitext(os.path.basename(filename))
	new_filepath = os.path.join(path, 'dev.txt')
	with open(new_filepath, 'w') as new_file:
		with open(filepath, 'r') as old_file:
			for line in old_file:
				items = line.strip().split(' , ')
				if len(items) == 2:
					text = items[-1].strip()
					label = items[0].strip()
				elif len(items) > 2:
					label = items[0].strip()
					text = ' , '.join(items[1:]).strip()
				else:
					continue
				new_line = text + '\t' + label + '\n'
				new_file.write(new_line)
	print('Finished')
