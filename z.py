 """ plt.pcolor(sg.w_input[:10])    
    plt.colorbar() 
    plt.show() """

    """ from sklearn.decomposition import PCA
    X = sg.vocabulary
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show() """