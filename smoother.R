#sink()
sink("smoother_output.txt", append=FALSE, split=FALSE)
smoother <- function(x,y,h) {
  meanclose <- function(t) 
    mean(y[abs(x-t) < h])
  sapply(x,meanclose)
}

x <- 1:50000
y <- 50001:100000
h <- 2
print(smoother(x, y, h))