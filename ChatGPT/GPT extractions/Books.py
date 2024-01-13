import random

class BooksChatbot:
    def __init__(self):
        self.user_preferences = {'genre': 'general', 'mood': '', 'interests': []}
        self.genre_recommendations = {
            'classic': ["Pride and Prejudice", "1984", "To Kill a Mockingbird"],
            'sci-fi': ["Dune", "The Martian", "Neuromancer"],
            'fantasy': ["The Hobbit", "Harry Potter and the Sorcerer's Stone", "The Name of the Wind"],
            'mystery': ["The Girl with the Dragon Tattoo", "And Then There Were None", "Gone Girl"],
        }
        self.trivia_questions = [
            "Which author wrote the famous line, 'It was the best of times, it was the worst of times'?",
            "What is the real name of the author George Orwell?",
            "In which year was 'To Kill a Mockingbird' published?"
        ]
        self.genres = ["fantasy", "science fiction", "mystery", "romance", "historical", "thriller"]
        self.favorite_quotes = [
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "It was the best of times, it was the worst of times."
        ]
        self.book_snack_pairings = {
            "fantasy": "Tea and scones",
            "science fiction": "Energy drink and protein bar",
            "mystery": "Coffee and a croissant",
            "romance": "Wine and chocolate",
            "historical": "Herbal tea and biscuits",
            "thriller": "Dark coffee and a bagel"
        }
        self.what_if_scenarios = [
            "What if Sherlock Holmes was in a futuristic world?",
            "Imagine Elizabeth Bennet with time-traveling abilities!",
            "What if Harry Potter was set in the steampunk era?"
        ]
        self.advanced_trivia = [
            "Who wrote the novel '1984'?",
            "What is the longest novel ever published?",
            "Which Shakespeare play features the character Puck?"
        ]

    def get_book_recommendation(self, genre):
        if genre.lower() in self.genres:
            return f"I recommend 'Sample Book' for a great {genre} read. Does this book meet your needs, or would you like me to recommend another?"
        else:
            return "I'm not sure about that genre. Can you specify another?"

    def discuss_genre(self, genre):
        if genre.lower() in self.genres:
            return f"{genre.title()} books are wonderful, aren't they? They transport you to different worlds and experiences!"
        else:
            return "That's an interesting genre! Tell me more about what you like in it."

    def literary_trivia(self):
        trivia = [
            "Who wrote 'Pride and Prejudice'?",
            "What is the real name of the author George Orwell?",
            "In which book does the character 'Holly Golightly' appear?"
        ]
        return random.choice(trivia)

    def quote_of_the_day(self):
        return random.choice(self.favorite_quotes)

    def suggest_snack_pairing(self, genre):
        return self.book_snack_pairings.get(genre.lower(), "I'm not sure about that pairing. How about a cup of tea and a good book?")

    def offer_what_if_scenario(self):
        return random.choice(self.what_if_scenarios)

    def challenge_with_advanced_trivia(self):
        return random.choice(self.advanced_trivia)

# Example usage
books_bot = BooksChatbot()
print(books_bot.get_book_recommendation("mystery"))
print(books_bot.discuss_genre("fantasy"))
print(books_bot.literary_trivia())
print(books_bot.quote_of_the_day())
print(books_bot.suggest_snack_pairing("romance"))
print(books_bot.offer_what_if_scenario())
print(books_bot.challenge_with_advanced_trivia())

    def update_preferences(self, genre, mood, interests):
        self.user_preferences = {'genre': genre, 'mood': mood, 'interests': interests}

    def recommend_book(self):
        genre = self.user_preferences['genre']
        recommendations = self.genre_recommendations.get(genre, ["The Catcher in the Rye"])
        recommended_book = random.choice(recommendations)

        return f"I recommend '{recommended_book}'. A splendid choice for a {genre} genre enthusiast! Does this book meet your needs, or would you like me to recommend another?"

    def literary_trivia(self):
        question = random.choice(self.trivia_questions)
        return question

    def discuss_book(self, book_title):
        # Enhanced discussion logic
        discussions = {
            "the great gatsby": "F. Scott Fitzgerald's 'The Great Gatsby' is a poignant exploration of the American Dream. What do you think about the character of Jay Gatsby?",
            "1984": "'1984' by George Orwell presents a dystopian future. How do you interpret its message about surveillance and freedom?"
        }
        return discussions.get(book_title.lower(), "I don't have information on that book, but I'd love to hear your thoughts on it!")

    def suggest_pairing(self, book_title):
        pairings = {
            "pride and prejudice": "A cup of tea and some scones would pair wonderfully with 'Pride and Prejudice'.",
            "the martian": "How about some freeze-dried snacks while reading 'The Martian'? It'll feel like you're right there with the protagonist!"
        }
        return pairings.get(book_title.lower(), "I don't have a specific pairing for that book, but a cozy blanket and your favorite beverage always make for a great reading experience!")

# Example of using the BooksChatbot
books_bot = BooksChatbot()
books_bot.update_preferences(genre='sci-fi', mood='adventurous', interests=['space', 'technology'])
print(books_bot.recommend_book())
print(books_bot.literary_trivia())
print(books_bot.discuss_book("1984"))
print(books_bot.suggest_pairing("The Martian"))
