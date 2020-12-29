def mayor_usac(usac1, usac2, usac3, usac4, usac5):
    maximo = max([usac1.test_accuracy,usac2.test_accuracy,usac3.test_accuracy,usac4.test_accuracy,usac5.test_accuracy])

    if usac1.test_accuracy==maximo:
        return usac1
    elif usac2.test_accuracy==maximo:
        return usac2
    elif usac3.test_accuracy==maximo:
        return usac3
    elif usac4.test_accuracy==maximo:
        return usac4
    else:
        return usac5

def mayor_mariano(mariano1, mariano2, mariano3, mariano4, mariano5):
    maximo = max([mariano1.test_accuracy,mariano2.test_accuracy,mariano3.test_accuracy,mariano4.test_accuracy,mariano5.test_accuracy])

    if mariano1.test_accuracy==maximo:
        return mariano1
    elif mariano2.test_accuracy==maximo:
        return mariano2
    elif mariano3.test_accuracy==maximo:
        return mariano3
    elif mariano4.test_accuracy==maximo:
        return mariano4
    else:
        return mariano5

def mayor_marroquin(marroquin1, marroquin2, marroquin3, marroquin4, marroquin5):
    maximo = max([marroquin1.test_accuracy,marroquin2.test_accuracy,marroquin3.test_accuracy,marroquin4.test_accuracy,marroquin5.test_accuracy])

    if marroquin1.test_accuracy==maximo:
        return marroquin1
    elif marroquin2.test_accuracy==maximo:
        return marroquin2
    elif marroquin3.test_accuracy==maximo:
        return marroquin3
    elif marroquin4.test_accuracy==maximo:
        return marroquin4
    else:
        return marroquin5

def mayor_landivar(landivar1, landivar2, landivar3, landivar4, landivar5):
    maximo = max([landivar1.test_accuracy,landivar2.test_accuracy,landivar3.test_accuracy,landivar4.test_accuracy,landivar5.test_accuracy])

    if landivar1.test_accuracy==maximo:
        return landivar1
    elif landivar2.test_accuracy==maximo:
        return landivar2
    elif landivar3.test_accuracy==maximo:
        return landivar3
    elif landivar4.test_accuracy==maximo:
        return landivar4
    else:
        return landivar5
        