class Rates(object):
    """
    Stores death, birth, and growth rates, as well as mutational effects.
    """
    def __init__(self, max_birth, death='0.1', bin_width='5e-4'):
        """
        The type of `max_birth` determines the type of all rates.
        
        The minimum birth rate is zero.
        """
        dtype = type(max_birth)
        self.death = dtype(death)
        self.bin_width = dtype(bin_width)
        self.n_classes = int(mp.nint(max_birth / self.bin_width)) + 1
        self.birth = np.array(linspace(0, max_birth, self.n_classes))
        self.fitness = self.birth - self.death
        self.growth = self.fitness
        self.effects = np.concatenate((-self.birth[::-1], self.birth[1:]))
        half_w = self.bin_width / 2
        walls = np.concatenate(([-half_w], self.birth + half_w))
        self.fitness_walls = walls - self.death