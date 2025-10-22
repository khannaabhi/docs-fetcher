import lancedb
import os
import shutil

class LanceDBManager:
    """
    Manages interactions with the LanceDB vector store.
    """
    def __init__(self, db_path: str, table_name: str):
        """
        Initializes the LanceDB manager.

        Args:
            db_path (str): The file system path where the LanceDB database will be stored.
            table_name (str): The name of the table within the LanceDB database.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.db = None # Will be initialized on first connection

    def _connect(self):
        """Internal method to establish connection to LanceDB."""
        if self.db is None:
            print(f"Connecting to LanceDB at: {self.db_path}")
            self.db = lancedb.connect(self.db_path)
        return self.db

    async def add_documents(self, documents: list[dict]):
        """
        Adds a list of documents (with text, id, source_path, and vector) to LanceDB.
        If the table doesn't exist, it will be created.
        """
        if not documents:
            print("No documents to add to LanceDB.")
            return

        self._connect() # Ensure connection is established

        try:
            if self.table_name in self.db.table_names():
                table = self.db.open_table(self.table_name)
                print(f"Table '{self.table_name}' already exists. Appending new data.")
                table.add(documents)
            else:
                table = self.db.create_table(self.table_name, data=documents)
                print(f"Table '{self.table_name}' created and data added.")
            print(f"Added {len(documents)} documents to LanceDB.")
        except Exception as e:
            print(f"Error storing data in LanceDB: {e}")
            # Re-raise to ensure main knows about the failure if critical
            raise

    async def get_table(self):
        """
        Returns the LanceDB table instance.
        """
        self._connect()
        if self.table_name not in self.db.table_names():
            print(f"Table '{self.table_name}' does not exist. Please add documents first.")
            return None
        return self.db.open_table(self.table_name)

    async def get_document_count(self):
        """
        Returns the total number of documents in the LanceDB table.
        """
        self._connect()
        if self.table_name not in self.db.table_names():
            return 0
        table = self.db.open_table(self.table_name)
        return table.count_rows()

    def clear_data(self):
        """
        Removes the LanceDB data directory for a fresh start.
        """
        if os.path.exists(self.db_path):
            print(f"Removing existing LanceDB directory: {self.db_path}")
            shutil.rmtree(self.db_path)
            print("Existing LanceDB directory removed.")
        else:
            print(f"LanceDB directory '{self.db_path}' does not exist. Nothing to remove.")

