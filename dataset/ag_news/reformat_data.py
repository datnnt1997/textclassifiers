import sys
import os
import pandas as pd

if __name__ == '__main__':
	filepath = "./test.csv"
	path, filename = os.path.split(filepath)
	name, ext = os.path.splitext(os.path.basename(filename))
	new_filepath = os.path.join(path, 'processed_test.csv')
	count = 0
	with open(new_filepath, 'w') as new_file:
		with open(filepath, 'r') as old_file:
			df = pd.read_csv(old_file)
			print(df.shape)
			for row in df.values:
				if len(row) == 3:
					text = row[-1].strip()
					label = row[0]
				else:
					continue
				new_line = text + '\t' + str(label) + '\n'
				new_file.write(new_line)
				count += 1

	print('Finished')
