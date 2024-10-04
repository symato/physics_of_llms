import lzma, json

most_uncompressed_vi_words = [ json.loads(line) for line in lzma.open("data/vi_words_score.jsonl.xz") ]

# print(most_uncompressed_vi_words)


if __name__ == "__main__":

	def impact(num_chosen_words):
		total_score = sum( [ x["score"] for x in most_uncompressed_vi_words[ : num_chosen_words] ] )
		# return total_score // num_chosen_words
		return total_score

	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np

	n = len(most_uncompressed_vi_words)
	impact = np.array([ impact(num_chosen_words) for num_chosen_words in range(1, n + 1) ])
	df = pd.DataFrame(list(zip(list(range(0, n)), impact)), columns=['words', 'impact'])
	print(df)

	plt.bar(df.words, df.impact)
	plt.xlabel('CHOSEN WORDS')
	plt.ylabel('TOTAL SCORES')
	# plt.yscale('log')
	plt.show()
