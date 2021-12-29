Rscript optuna.R
cd stat.model
pdfcrop optuna.db.dptb.pdf optuna.db.dptb.pdf
pdfcrop optuna.db.tiger.pdf optuna.db.tiger.pdf
pdfcrop optuna.dm.dptb.pdf optuna.dm.dptb.pdf
pdfcrop optuna.dm.tiger.pdf optuna.dm.tiger.pdf
pdfcrop diff.optuna.dm.tiger.pdf diff.optuna.dm.tiger.pdf
mv optuna.db.dptb.pdf ~/KK/TACL_DCCP/figures
mv optuna.db.tiger.pdf ~/KK/TACL_DCCP/figures
mv optuna.dm.dptb.pdf ~/KK/TACL_DCCP/figures
mv optuna.dm.tiger.pdf ~/KK/TACL_DCCP/figures
mv diff.optuna.dm.tiger.pdf ~/KK/TACL_DCCP/figures
