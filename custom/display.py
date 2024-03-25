import numpy as np
from pymoo.util.display.column import Column
from pymoo.util.display.multi import MultiObjectiveOutput


class moo_display(MultiObjectiveOutput):
    def __init__(self):
        super().__init__()
        self.ave_precision = Column("average precision (f1)", width=25)
        self.ave_recall = Column("average recall (f2)", width=25)
        self.best_precision = Column("best precision (f1)", width=20)
        self.best_recall = Column("best recall (f2)", width=20)
        self.train_best_f1 = Column("best train f1-score", width=20)
        self.train_ave_f1 = Column("ave train f1-score", width=20)
        self.val_best_f1 = Column("best val f1-score", width=20)
        self.val_ave_f1 = Column("ave val f1-score", width=20)

    def initialize(self, algorithm):
        super().initialize(algorithm)
        self.columns += [
            self.best_precision,
            self.ave_precision,
            self.best_recall,
            self.ave_recall,
            self.train_best_f1,
            self.train_ave_f1,
            self.val_best_f1,
            self.val_ave_f1,
        ]

    def update(self, algorithm):
        super().update(algorithm)
        self.ave_precision.set(
            f"{np.mean( 1 -algorithm.opt.get('F')[:, 0]):.6f} (\u00B1{np.std( 1 -algorithm.opt.get('F')[:, 0]):.6f})"
        )
        self.ave_recall.set(
            f"{np.mean( 1 -algorithm.opt.get('F')[:, 1]):.6f} (\u00B1{np.std( 1 -algorithm.opt.get('F')[:, 1]):.6f})"
        )
        self.best_precision.set(
            f"{np.max( 1 -algorithm.opt.get('F')[:, 0]):.6f}"
        )
        self.best_recall.set(
            f"{np.max( 1 -algorithm.opt.get('F')[:, 1]):.6f}"
        )

        if algorithm.n_iter % 5 == 0 or algorithm.n_iter == 1:
            pf_val, pf_train = [], []
            pf = algorithm.opt.get("X")
            pf = algorithm.unblocker(pf)
            # print(pf[:5])
            for i in range(pf.shape[0]):
                val = algorithm.test_validator(pf[i])
                pf_val.append(val)
                val = algorithm.train_validator(pf[i])
                pf_train.append(val)

            self.train_best_f1.set(f"{np.max(pf_train):.6f}")
            self.train_ave_f1.set(f"{np.mean(pf_train):.6f}")
            self.val_best_f1.set(f"{np.max(pf_val):.6f}")
            self.val_ave_f1.set(f"{np.mean(pf_val):.6f}")

        else:
            self.train_best_f1.set(f"-")
            self.train_ave_f1.set(f"-")
            self.val_best_f1.set(f"-")
            self.val_ave_f1.set(f"-")