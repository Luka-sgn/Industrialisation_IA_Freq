import numpy as np

class FreqPredictor:
    def __init__(self):
        self.model = None
        self.encoder = None

    def predict(self, df, num_cols, cat_cols):
        df_encoded = self.encoder.transform(df.copy())
        return self.model.predict(df_encoded)

    @staticmethod
    def reduce_memory_usage(df, verbose=True):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min >= 0:
                        if c_max < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif c_max < 65535:
                            df[col] = df[col].astype(np.uint16)
                        else:
                            df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.float32)
        return df
