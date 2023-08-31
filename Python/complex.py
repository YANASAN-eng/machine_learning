import math
#複素数
class Complex:
    real = 0
    image = 0
    def __init__(self,real,image):
        self.real = real
        self.image = image
    @classmethod
    def add(cls,z_1,z_2):
        z = Complex(z_1.real + z_2.real,z_1.image + z_2.image)
        return z
    @classmethod
    def minus(cls,z):
        w = Complex(0,0)
        w.real = -z.real
        w.image = -z.image
        return w
    @classmethod
    def times(cls,z_1,z_2):
        z = Complex(z_1.real * z_2.real - z_1.image * z_2.image,z_1.real * z_2.image + z_1.image * z_2.real)
        return z
    @classmethod
    def norm(cls,z):
        return math.sqrt(z.real * z.real + z.image * z.image)
    @classmethod
    def reciprocal(cls,z):
        w = Complex(0,0)
        w.real = z.real / (cls.norm(z)**2)
        w.image = -z.image / (cls.norm(z)**2)
        return w
    @classmethod
    def conjugate(cls,z):
        w = Complex(0,0)
        w.real = z.real
        w.image = -z.image
        return w
    @classmethod
    def argument(cls,z):
        w = Complex(0,0)
        w.real = z.real / cls.norm(z)
        w.image = z.image / cls.norm(z)
        return w
    @classmethod
    def output(cls,z):
        print(str(format(z.real,'.5f')) + " + i(" + str(format(z.image,'.5f')) + ")",end = " ")