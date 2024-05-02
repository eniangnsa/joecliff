import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')

# create an evaluator for text complexity       
class TextEvaluator:
    def __init__(self, text):
        self.text = text
        
    def sent_counter(self):
        '''
        This is a function to count the number of sentences in a given text
        '''
        # Split text into sentences using typical punctuation marks as delimiters
        sentences = self.text.split('.')

        # Remove empty strings and strip whitespace from each sentence
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return len(sentences)
    
    
    def no_words(self):
        '''
        Function to count and return the number of words in a text
        '''
        # Split text into words using whitespace as delimiter
        words = self.text.split()
        # Remove empty strings and strip punctuation from each word
        words = [word.strip(",.!?;:()«»") for word in words if word.strip(",.!?;:()«»")]
        return len(words)
    
    
    def no_syllables(self):
        '''
        Counts the number of syllables in a word
        '''
        # Here, we use a simple heuristic approach based on the number of vowels in each word
        vowels = "аеёиоуыэюя"
        syllable_count = 0

        # Split text into words using whitespace as delimiter
        words = self.text.split()

        for word in words:
            # Count the number of vowels in each word
            syllable_count += sum(1 for letter in word if letter.lower() in vowels)

        return syllable_count
    
    def ASW(self):
        '''
        This computes the average number of syllabus in a given text
        
        asw : Average syllabus length
        nw        : number of words
        nse       : number of syllabus
        
        '''
        # Calculate the average number of syllabus
        self.no_sentence = self.sent_counter() # number of sentence in a text
        self.nw = self.no_words() # number of words in the text
        self.nsy = self.no_syllables() # number of syllables in text
        if self.nsy != 0:
            self.asw = self.nw / self.nsy
        else:
            self.asw = 0  # Handle division by zero
        return self.asw
    
    
    # function to calculate average sentence length
    def ASL(self):
        '''
        computes the average sentence length.
        
        DESCRIPTION:
        
        asl : average length of the sentence.
        no_sentence : number of sentence in the text
        '''
        if self.no_sentence != 0:
            self.asl = self.nw / self.no_sentence
        else:
            self.asl = 0
        return self.asl
    
    
    def calculate_ttr(self, tokens):
        '''
        ttr stands for type token ratio.
        DESCRIPTION:
        total_tokens : returns the total number of tokens
        total_types  : returns a set with all the types of tokens
        '''
        total_tokens = len(tokens)
        total_types = len(set(tokens))
        return total_types / total_tokens if total_tokens > 0 else 0

    def TTR(self):
        '''
        Computes the TTR.
        words : a list of russian words that have been tokenized from the text
        '''
        # Tokenize the text into words
        words = word_tokenize(self.text, language='russian')
        return self.calculate_ttr(words)

    def TTR_N(self):
        '''
        TTR_N  stands for type-token ratio for Nouns only.
        '''
        # Tokenize the text into words and perform part-of-speech tagging
        words = word_tokenize(self.text, language='russian')
        tagged_words = pos_tag(words, lang='rus')
        
        # Filter nouns
        nouns = [word for word, tag in tagged_words if tag.startswith('S')]
        return self.calculate_ttr(nouns)

    def TTR_V(self):
        '''
        TTR_V stands for type-token ratio for verbs only.
        '''
        # Tokenize the text into words and perform part-of-speech tagging
        words = word_tokenize(self.text, language='russian')
        tagged_words = pos_tag(words, lang='rus')
        
        # Filter verbs
        verbs = [word for word, tag in tagged_words if tag.startswith('V')]
        return self.calculate_ttr(verbs)

    def TTR_A(self):
        '''
        TTR_A stands for type-token ratio for Adjectives only.
        '''
        # Tokenize the text into words and perform part-of-speech tagging
        words = word_tokenize(self.text, language='russian')
        tagged_words = pos_tag(words, lang='rus')
        
        # Filter adjectives
        adjectives = [word for word, tag in tagged_words if tag.startswith('A')]
        return self.calculate_ttr(adjectives)

    def NAV(self):
        ttr_a = self.TTR_A()
        ttr_n = self.TTR_N()
        ttr_v = self.TTR_V()
        
        if ttr_v != 0:
            return (ttr_a + ttr_n) / ttr_v
        else:
            return 0  # Handle division by zero
    
    
    def UNAV(self):
        # Tokenize the text into words
        words = word_tokenize(self.text, language='russian')
        
        # Perform part-of-speech tagging
        tagged_words = pos_tag(words, lang='rus')
        
        # Count unique nouns, adjectives, and verbs
        nouns = set()
        adjectives = set()
        verbs = set()
        for word, tag in tagged_words:
            if tag.startswith('S'):  # Noun
                nouns.add(word.lower())
            elif tag.startswith('A'):  # Adjective
                adjectives.add(word.lower())
            elif tag.startswith('V'):  # Verb
                verbs.add(word.lower())
        
        # Calculate UNAV metric
        unique_nouns_adjectives = len(nouns.union(adjectives))
        unique_verbs = len(verbs)
        
        if unique_verbs != 0:
            return unique_nouns_adjectives / unique_verbs
        else:
            return 0  # Handle division by zero

    

    def corpus_table(self):
        
        self.table = {'nw': {self.nw}, 'nsy':{self.nsy}, 'ASL':{self.asl}, 'ASW':{self.asw}}
        return self.table
    
    # metrics for evaluating the russian text
    def m3(self):    
        # create variables for unav that will be later used to make computation
        self.unav = self.UNAV()
        return -9.53 + 0.25 * self.asl + 4.98 * self.asw + 0.89 * self.unav
    
     # Flesch Reading Ease
    def FRE(self):
        ''''
        Computes the Flesch Reading Ease score. 
        DESCRIPTION:
        asl : average sentence length
        asw : average syllable length
        '''
        return 206.835 - 1.015 * self.asl - 84.6 * self.asw
      
    def q_funct(self):
        nav = self.NAV()
        return -0.124*self.asl + 0.018*self.asw - 0.007*self.unav - 0.003*self.asl**2 + 0.184*self.asl*self.asw + 0.097*self.asl*self.unav - 0.158*self.asl*nav + 0.09*self.asw**2 + 0.091*self.asw*self.unav + 0.023*self.asw*nav - 0.157*self.unav**2 - 0.079*self.unav*nav + 0.058*nav**2
    
    def report(self):
        self.quad = self.q_funct()
        self.M3 = self.m3()
        self.fre = self.FRE()
        self.result = {'Q_metric':{self.quad}, 'M3':{self.M3}, 'FRE':{self.fre}}
        return self.result
        
        
# Sample text in Russian
sample_text = "В данной работе хотелось бы остано-виться подробнее на анализе текста в пла-не категории оценки, так как, с одной сто-роны, эта категория нередко оказывается вне внимания исследователей (так, напри-мер, в монографии А.Ф.Папиной10 оценка называется «пятой глобальной категори-ей» после времени, художественного про-странства и др., а в замечательной книге К.А.Роговой и коллектива её соавторов11 категория оценки даже не упоминает-ся, хотя категории интенциональность, цельность,  связность, информативность, воспринимаемость,  ситуативность и интертекстуальность — освещены ос-новательно), с другой стороны, именно частнооценочные оппозиции (нормаль-но — аномально, важно — неважно) и их языковое выражение входят в сферу на-ших научных интересов [1; 2; 3]. Кроме того, по нашему глубокому убеждению, именно в оценке наиболее явно проступа-ет (языковая) личность автора и становит-ся понятной авторская концепция текста"

# Create an instance of TextEvaluator
evaluator = TextEvaluator(sample_text)

# Test sentence_counter method
print("Number of sentences:", evaluator.sent_counter())

# Test no_words method
print("Number of words:", evaluator.no_words())

# Test no_syllables method
print("Number of syllables:", evaluator.no_syllables())

# Test ASW method
print("Average syllables per word:", evaluator.ASW())

# Test ASL method
print("Average sentence length:", evaluator.ASL())

# Test TTR method
print("Type-token ratio for all tokens:", evaluator.TTR())

# Test TTR_N method
print("Type-token ratio for nouns only:", evaluator.TTR_N())

# Test TTR_V method
print("Type-token ratio for verbs only:", evaluator.TTR_V())

# Test TTR_A method
print("Type-token ratio for adjectives only:", evaluator.TTR_A())

# Test NAV method
print("NAV:", evaluator.NAV())

# Test UNAV method
print("UNAV:", evaluator.UNAV())

# Test m3 method
print("m3:", evaluator.m3())

# Test FRE method
print("FRE:", evaluator.FRE())

# Test q_funct method
print("Q metric:", evaluator.q_funct())

# Test report method
print("Text complexity report:", evaluator.report())

    
 