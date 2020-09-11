import os

#
FOLDER = 'vids'
vidlist = []
for root, dirnames, filenames in os.walk(FOLDER):
	for filename in filenames:
		if filename.endswith('.mp4'):
			vidlist.append(os.path.join(root, filename))
		else:
			print(filename)

#
f = open('list.txt', 'w')
for vid in vidlist:
	f.write("file '" + vid + "'\n")
f.close()

#
cmd = 'ffmpeg -f concat -i list.txt -c copy concat.mp4'
os.system(cmd)

os.system('rm list.txt')