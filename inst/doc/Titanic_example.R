## ----setup, echo=FALSE, cache=FALSE-------------------------------------------
library(knitr)
library(rmdformats)

## Global options
options(max.print="75")
opts_chunk$set(echo=TRUE,
	             cache=FALSE,
               prompt=FALSE,
               tidy=FALSE,
               comment=NA,
               message=FALSE,
               warning=FALSE)
opts_knit$set(width=75)

## ----load_res, include=FALSE--------------------------------------------------
load(url('http://nicolas.robette.free.fr/Docs/results_titanic.RData'))

## ----init, cache=FALSE--------------------------------------------------------
library(tidyverse)  # data management
library(caret)  # confusion matrix
library(party)  # conditional inference random forests and trees
library(partykit)  # conditional inference trees
library(pROC)  # ROC curves
library(measures)  # performance measures
library(varImp)  # variable importance
library(pdp)  # partial dependence
library(vip)  # measure of interactions
library(moreparty)  # surrogate trees, accumulated local effects, etc.
library(RColorBrewer)  # color palettes
library(GDAtools)  # bivariate analysis

## ----import_tita--------------------------------------------------------------
data(titanic)
str(titanic)

## ----desc_tita----------------------------------------------------------------
summary(titanic)

## ----bivar_assoc--------------------------------------------------------------
BivariateAssoc(titanic$Survived, titanic[,-1])

## ----catdesc------------------------------------------------------------------
catdesc(titanic$Survived, titanic[,-1], min.phi=0.1)

## ----seed---------------------------------------------------------------------
set.seed(1912)

## ----ctree, out.width='100%'--------------------------------------------------
arbre <- partykit::ctree(Survived~., data=titanic, control=partykit::ctree_control(minbucket=30, maxsurrogate=Inf, maxdepth=3))

print(arbre)

plot(arbre)

## ----proba_nodes--------------------------------------------------------------
nodeapply(as.simpleparty(arbre), ids = nodeids(arbre, terminal = TRUE), FUN = function(x) round(prop.table(info_node(x)$distribution),3))

## ----ctree_plot, out.width='100%'---------------------------------------------
plot(arbre, inner_panel=node_inner(arbre,id=FALSE,pval=FALSE), terminal_panel=node_barplot(arbre,id=FALSE), gp=gpar(cex=0.6), ep_args=list(justmin=15))

## ----pred_tree----------------------------------------------------------------
pred_arbre <- predict(arbre, type='prob')[,'Yes']

auc_arbre <- AUC(pred_arbre, titanic$Survived, positive='Yes')
auc_arbre %>% round(3)

## ----roc_tree, fig.align="center", fig.width=4, fig.height=4------------------
pROC::roc(titanic$Survived, pred_arbre) %>% 
  ggroc(legacy.axes=TRUE) +
    geom_segment(aes(x=0,xend=1,y=0,yend=1), color="darkgrey", linetype="dashed") +
    theme_bw() +
    xlab("TFP") +
    ylab("TVP")

## ----confusion----------------------------------------------------------------
ifelse(pred_arbre > .5, "Yes", "No") %>%
  factor %>%
  caret::confusionMatrix(titanic$Survived, positive='Yes')

## ----split_stats--------------------------------------------------------------
GetSplitStats(arbre)

## ----forest-------------------------------------------------------------------
foret <- party::cforest(Survived~., data=titanic, controls=party::cforest_unbiased(mtry=2,ntree=500))

## ----forest_pred--------------------------------------------------------------
pred_foret <- predict(foret, type='prob') %>%
              do.call('rbind.data.frame',.) %>%
              select(2) %>%
              unlist

auc_foret <- AUC(pred_foret, titanic$Survived, positive='Yes')
auc_foret %>% round(3)

## ----forest_pred_OOB----------------------------------------------------------
pred_oob <- predict(foret, type='prob', OOB=TRUE) %>%
              do.call('rbind.data.frame',.) %>%
              select(2) %>%
              unlist

auc_oob <- AUC(pred_oob, titanic$Survived, positive='Yes')
auc_oob %>% round(3)

## ----surrogate, out.width='100%'----------------------------------------------
surro <- SurrogateTree(foret, maxdepth=3)

surro$r.squared %>% round(3)

plot(surro$tree, inner_panel=node_inner(surro$tree,id=FALSE,pval=FALSE), terminal_panel=node_boxplot(surro$tree,id=FALSE), gp=gpar(cex=0.6), ep_args=list(justmin=15))

## ----vimp, fig.align="center", fig.width=5, fig.height=3----------------------
importance <- -varImpAUC(foret)
importance %>% round(3)

ggVarImp(importance)

## ----pdp, eval=FALSE----------------------------------------------------------
#  pdep <- GetPartialData(foret, which.class=2, probs=1:19/20, prob=TRUE)

## ----pdp2---------------------------------------------------------------------
pdep

## ----pdp_plot, fig.align="center", fig.width=5, fig.height=6------------------
ggForestEffects(pdep, vline=mean(pred_foret), xlab="Probability of survival") +
  xlim(c(0,1))

## ----pdp_age, eval=FALSE------------------------------------------------------
#  pdep_age <- pdp::partial(foret, 'Age', which.class=2, prob=TRUE, quantiles=TRUE, probs=1:39/40)

## ----pdp_plot_age, fig.align="center", fig.width=5, fig.height=3--------------
ggplot(pdep_age, aes(x=Age, y=yhat)) +
  geom_line() +
  geom_hline(aes(yintercept=mean(pred_foret)), size=0.2, linetype='dashed', color='black') +
  ylim(c(0,1)) +
  theme_bw() +
  ylab("Probability of survival")

## ----pdp_in, eval=FALSE-------------------------------------------------------
#  pdep_ind <- GetPartialData(foret, which.class=2, probs=1:19/20, prob=TRUE, ice=TRUE)

## ----pdp_table----------------------------------------------------------------
pdep_ind %>% group_by(var, cat) %>% summarise(prob = mean(value) %>% round(3),
                                              Q1 = quantile(value, 0.25) %>% round(3),
                                              Q3 = quantile(value, 0.75) %>% round(3))

## ----pdp_boxplot, fig.align="center", fig.width=5, fig.height=5---------------
ggplot(pdep_ind, aes(x = value, y = cat, group = cat)) + 
         geom_boxplot(aes(fill=var), notch=TRUE) + 
         geom_vline(aes(xintercept=median(pred_foret)), size=0.2, linetype='dashed', color='black') +
         facet_grid(var ~ ., scales = "free_y", space = "free_y") + 
         theme_bw() + 
         theme(panel.grid = element_blank(),
               panel.grid.major.y = element_line(size=.1, color="grey70"),
               legend.position = "none",
               strip.text.y = element_text(angle = 0)) +
         xlim(c(0,1)) +
         xlab("Probability of survival") +
         ylab("")

## ----ale, eval=FALSE----------------------------------------------------------
#  ale <- GetAleData(foret)

## ----ale2---------------------------------------------------------------------
ale

## ----ale_plot, fig.align="center", fig.width=5, fig.height=6------------------
ggForestEffects(ale)

## ----vint, eval=FALSE---------------------------------------------------------
#  vint <- GetInteractionStrength(foret)

## ----vint2--------------------------------------------------------------------
vint

## ----pd_inter2_sexclass, eval=FALSE-------------------------------------------
#  pdep_sexclass <- pdp::partial(foret, c('Sex','Pclass'), quantiles=TRUE, probs=1:19/20, which.class=2L, prob=TRUE)

## ----pd_plot_inter2_sexclass, fig.align="center", fig.width=5, fig.height=3----
ggplot(pdep_sexclass, aes(Pclass, yhat)) +
  geom_point(aes(color=Sex)) +
  ylim(0,1) +
  theme_bw()

## ----pd_inter2_sexage, eval=FALSE---------------------------------------------
#  pdep_sexage <- pdp::partial(foret, c('Sex','Age'), quantiles=TRUE, probs=1:19/20, which.class=2L, prob=TRUE)

## ----pd_plot_inter2_sexage, fig.align="center", fig.width=5, fig.height=3-----
ggplot(pdep_sexage, aes(Age, yhat)) +
  geom_line(aes(color=Sex)) +
  ylim(0,1) +
  theme_bw()

## ----pd_inter3, eval=FALSE----------------------------------------------------
#  pdep_sexclassage <- pdp::partial(foret, c('Sex','Pclass','Age'), quantiles=TRUE, probs=1:19/20, which.class=2L, prob=TRUE)

## ----pd_plot_inter3, eval=FALSE-----------------------------------------------
#  cols <- c(paste0('dodgerblue',c(4,3,1)),paste0('tomato',c(4,3,1)))
#  pdep_sexclassage %>% mutate(sexclass = interaction(Pclass,Sex)) %>%
#                       ggplot(aes(x=Age, y=yhat)) +
#                         geom_line(aes(colour=sexclass)) +
#                         scale_color_manual(values=cols) +
#                         ylim(0,1) +
#                         theme_bw()

## ----pd_plot_inter3bis, echo=FALSE, fig.align="center", out.width='70%'-------
knitr::include_graphics("http://nicolas.robette.free.fr/Docs/plot_inter3.png")

## ----ale_inter2, eval=FALSE---------------------------------------------------
#  ale_sex_age = GetAleData(foret, xnames=c("Sex","Age"), order=2)

## ----ale_plot_inter2, fig.align="center", fig.width=5, fig.height=3-----------
ale_sex_age %>% ggplot(aes(Age, value)) + 
                  geom_line(aes(color=Sex)) +
                  geom_hline(yintercept=0, linetype=2, color='gray60') +
                  theme_bw()

## ----prototypes---------------------------------------------------------------
prox <- proximity(foret)
proto <- Prototypes(titanic$Survived, titanic[,-1], prox)
proto

## ----outliers1, fig.align="center", fig.width=4, fig.height=4-----------------
out <- bind_cols(pred=round(pred_foret,2),titanic) %>%
         Outliers(prox, titanic$Survived, .)
boxplot(out$scores)

## ----outliers2----------------------------------------------------------------
arrange(out$outliers, Survived, desc(scores)) %>%
  split(.$Survived)

## ----featsel, eval=FALSE------------------------------------------------------
#  featsel <- FeatureSelection(titanic$Survived, titanic[,-1], method="RFE", positive="Yes")

## ----featsel2-----------------------------------------------------------------
featsel$selection.0se
featsel$selection.1se

## ----parallel, results='hold'-------------------------------------------------
library(doParallel)
registerDoParallel(cores=2)
fastvarImpAUC(foret)
stopImplicitCluster()

