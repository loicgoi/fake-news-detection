from typing import final
import os
import pandas as pd
from data_handler.text_cleaning import text_cleaning


@final
class DataHandler:
    def __init__(self, csv_path: str):
        self.original_csv_path = csv_path
        
        # Si le chemin Docker n'existe pas, essayer le chemin local
        if csv_path.startswith('/app/') and not os.path.exists(csv_path):
            # Convertir /app/data/... en ./data/...
            local_path = csv_path.replace('/app/', './')
            if os.path.exists(local_path):
                self.csv_path = local_path
                print(f"✓ Using local path: {local_path}")
            else:
                self.csv_path = csv_path
        else:
            self.csv_path = csv_path

    def load(self):
        print(f"· LOADING CSV DATA FROM `{self.csv_path}` INTO A DATAFRAME")
        
        # Vérifier que le fichier existe
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        print(f"· DATAFRAME PREVIEW:\n{self.df.head()}")
        return self

    @staticmethod
    def explore(df):
        print("· EXPLORING DATA")
        if df is None:
            raise ValueError("× Dataframe not found")
        exploration_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isna().sum().to_dict(),
        }
        print(f"· SHAPE: {exploration_info['shape']}")
        print(f"· COLUMNS: {exploration_info['columns']}")
        print(f"· DTYPES: {exploration_info['dtypes']}")
        print(f"· MISSING: {exploration_info['missing']}")
        return df

    @staticmethod
    def clean(df):
        print("· CLEANING DATA")
        clean_df: pd.DataFrame = df.copy()

        clean_df = clean_df.drop_duplicates("text")
        print(f"· DROPPED {df['text'].duplicated().sum()} DUPLICATES FOR `text` COLUMN")

        for column in ["title", "text", "subject"]:
            clean_df[column] = clean_df[column].apply(text_cleaning)
        print("· DECODED HTML ENTITIES BACK TO THEIR ORIGINAL CHARACTERS")
        print("· REMOVED HTML TAGS")
        print("· REMOVED OTHER UNWANTED FORMATTING TAGS")
        print("· REMOVED URLS")
        print("· NORMALIZED WHITE SPACES")
        print("· REMOVED SPECIAL CARACTERS")

        clean_df["date"] = pd.to_datetime(
            clean_df["date"], errors="coerce", format="mixed"
        )
        print("· CONVERTED `date` COLUMN TO DATETIME FORMAT")

        clean_df = clean_df[clean_df["title"].str.strip().astype(bool)]  # pyright: ignore[reportAssignmentType]
        clean_df = clean_df[clean_df["text"].str.strip().astype(bool)]  # pyright: ignore[reportAssignmentType]
        print("· DELETED ENTRIES WITH EMPTY `title` OR `text` VALUES")

        clean_df = clean_df.reset_index(drop=True)
        print("· RESET INDEX")

        print(f"· CLEAN DATAFRAME PREVIEW:\n{clean_df}")
        return clean_df
