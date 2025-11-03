import re

import pandas as pd
import questionary
from chroma.chroma_manager import ChromaManager
from data_handler.data_handler import DataHandler
from data_handler.text_cleaning import text_cleaning
from prompt.rag_system import RAGSystem


class Pipeline:
    def __init__(self):
        self.handlers = {
            "fake_news_csv": DataHandler("/app/data/Fake.csv"),
            "true_news_csv": DataHandler("/app/data/True.csv"),
        }
        self._is_loaded = False
        self._is_clean = False
        self._is_user_input = False

    def data_loading(self):
        """
        Loops in handlers dict to call their `load()` method.

        Parameters:
            self

        Returns:
            self
        """
        fake_df = self.handlers["fake_news_csv"].load().df
        true_df = self.handlers["true_news_csv"].load().df
        fake_df["label"] = "fake"
        true_df["label"] = "true"

        self.df = pd.concat(
            [
                fake_df,
                true_df,
            ],
            ignore_index=True,
        )
        self._is_loaded = True
        return self

    def data_exploration(self):
        """
        Loops in handlers dict to call their `explore()` method.

        Parameters:
            self

        Returns:
            self
        """
        if not self._is_loaded:
            print("× DATA NEEDS TO BE LOADED, LOADING DATA...")
            self.data_loading()
        print("\n-> DATA EXPLORATION STARTING...")
        self.df = DataHandler.explore(self.df)
        return self

    def data_cleaning(self):
        """
        Loops in handlers dict to call their `clean()` method.

        Parameters:
            self

        Returns:
            self
        """
        if not self._is_loaded:
            print("× DATA NEEDS TO BE LOADED, LOADING DATA...")
            self.data_loading()
        print("\n-> DATA CLEANING STARTING...")
        self.df = DataHandler.clean(self.df)
        self._is_clean = True
        return self

    def chroma_insertion(self):
        """
        Creates chunks, embeds and normalizes them, then inserts into chromaDB

        Parameters:
            self

        Returns:
            self
        """
        if not self._is_clean:
            print("× DATA NEEDS TO BE CLEAN, CLEANING DATA...")
            self.data_cleaning()
        chroma = ChromaManager("news")
        print("\n-> PROCESSING DATA AND INSERTING IT INTO CHROMADB")
        chroma.add_dataframe_to_collection(self.df, "news")
        return self

    def process_user_input(self):
        user_input: str = input("Please provide the body of a news article: ")
        clean_user_input = text_cleaning(user_input)
        rag_system = RAGSystem()
        self.llm_response = rag_system.analyze_article(clean_user_input)
        print(self.llm_response)
        self.evaluation = rag_system.evaluation_rag()

        print(f"\nscore: {self.evaluation} %")
        return self

    def ask_user(self):
        is_first_time = True
        wish_to_continue = True
        while wish_to_continue:
            selection = questionary.select(
                (
                    "What do you want to do?"
                    if is_first_time
                    else "And now, what do you want to do?"
                ),
                [
                    "EXPLORATION -> Quickly load data from CSV files to take a look at the data exploration",
                    "INSERTION -> Load data from CSV files, clean it, process it and insert it into a chromaDB (might take a while)",
                    "RUN -> Provide the body of a news article and check whether or not it is reliable",
                    "EXIT -> Exit this program",
                ],
            ).ask()
            is_first_time = False
            regex_match = re.search(r"^([A-Z]+)\s->", str(selection))
            if regex_match:
                first_selection = regex_match.group(1)
            match first_selection:
                case "EXPLORATION":
                    self.data_exploration()
                case "INSERTION":
                    self.chroma_insertion()
                case "RUN":
                    self.process_user_input()
                case "EXIT":
                    print("Program exited")
                    return
            wish_to_continue = questionary.confirm(
                "Would you like to do something else?"
            ).ask()
