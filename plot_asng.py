import matplotlib.pyplot as plt

f = open('models/results.txt', 'r')

asng = [float(line) for line in f.readlines()]
episodes = range(0, 14500, 100)


plt.figure(figsize=(9, 3))
plt.plot(episodes, asng)
plt.hlines(y=30, xmin=0, xmax=14500, linewidth=2, color='r', linestyle='--')
plt.xlabel('episodes')
plt.ylabel('ASNG metric')
plt.title('Average sheep near goal (ASNG) for last 100 episodes during training')
plt.savefig('asng_train.png', bbox_inches="tight")
plt.show()

