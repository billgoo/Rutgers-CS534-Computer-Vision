import PatchBasedSynthesis as pbs

if __name__ == "__main__":
    filename = ["T1.jpg", "T2.jpg", "T3.jpg", "T4.jpg", "T5.jpg"]
    for f in filename:
        # patch size = 25
        arg = [0, f, 25, 5, 0.3]
        pbs.main(arg)