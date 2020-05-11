import pandas as pd

class CommentsRepository:
    storage = None

    def __init__(self, storage):
        self.storage = storage

    def create_table(self):
        return self.storage.create_table()

    def find_parent_comment(self, pid):
        return self.storage.find_parent_comment(pid)

    def find_existing_score(self, pid):
        return self.storage.find_existing_score(pid)

    def replace_comment(self, comment_id, parent_id, parent_data, body, subreddit, created_at, score):
        return self.storage.replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_at, score)

    def insert_has_parent(self, comment_id, parent_id, parent_data, body, subreddit, created_at, score):
        return self.storage.insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_at, score)

    def insert_no_parent(self, comment_id, parent_id, body, subreddit, created_at, score):
        return self.storage.insert_no_parent(comment_id, parent_id, body, subreddit, created_at, score)

    def get_batch(self, unix, limit):
        return self.storage.get_batch(unix, limit)