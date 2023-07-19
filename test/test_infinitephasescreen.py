from aotools.turbulence import infinitephasescreen

def testVKInitScreen():

    scrn = infinitephasescreen.PhaseScreenVonKarman(128, 4./64, 0.2, 50, n_columns=4)

def testVKAddRow():

    scrn = infinitephasescreen.PhaseScreenVonKarman(128, 4./64, 0.2, 50, n_columns=4)
    scrn.add_row()



# Test of Kolmogoroc screen
def testKInitScreen():

    scrn = infinitephasescreen.PhaseScreenKolmogorov(128, 4./64, 0.2, 50, stencil_length_factor=4)

def testKAddRow():

    scrn = infinitephasescreen.PhaseScreenKolmogorov(128, 4./64, 0.2, 50, stencil_length_factor=4)
    scrn.add_row()

if __name__ == "__main__":

    from matplotlib import pyplot

    screen = infinitephasescreen.PhaseScreenVonKarman(64, 8./32, 0.2, 40, 2)

    pyplot.ion()
    pyplot.imshow(screen.stencil)

    pyplot.figure()
    pyplot.imshow(screen.scrn)
    pyplot.colorbar()
    for _ in range(100):
        screen.add_row()

        pyplot.clf()
        pyplot.imshow(screen.scrn)
        pyplot.colorbar()
        pyplot.draw()
        pyplot.pause(0.01)
