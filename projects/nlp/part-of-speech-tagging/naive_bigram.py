def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
        
    def get_bigrams_from_sequence(sequence):
        bigrams = []
        sequence_length = len(sequence)
        for i in range(sequence_length):
            if i < sequence_length - 1:
                bigrams.append((sequence[i], sequence[i+1]))
                
        return bigrams
    
    def count_sequence(sentence, sequence):
        # Initialize a counter
        count = 0

        # Get the length of the sequence
        win_size = len(sequence)

        # Iterate over the sentence with a sliding window
        for i in range(len(sentence) - win_size + 1):
            # Check if the slice of the sentence matches the sequence
            a = tuple(sentence[i:i + win_size])
            b = tuple(sequence)
            if tuple(sentence[i:i + win_size]) == tuple(sequence):
                count += 1

        return count

    result = {}
    
    unique_bigrams = {}    
    for sequence in sequences:
        bigrams = get_bigrams_from_sequence(sequence)
            
        for bigram in bigrams:
            if bigram not in unique_bigrams:
                unique_bigrams[bigram] = 0                        
        
    for unique_bigram in unique_bigrams:
        if unique_bigram not in result:
            result[unique_bigram] = 0
            
        for sequence in sequences:            
            result[unique_bigram] += count_sequence(sequence, unique_bigram)
    
    return result