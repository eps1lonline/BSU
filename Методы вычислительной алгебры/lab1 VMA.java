import java.text.DecimalFormat;

public class Main {
    public static void main(String[] args) {
        try {
            double[][] mtrA = {{0.3857, -0.0508, 0.0102, 0.0203, 0.0711},
                    {0.0528, 0.6039, 0.0000, -0.0406, 0.0406},
                    {0.0305, 0.0000, 0.4852, -0.1421, 0.0812},
                    {-0.0609, 0.1279, 0.0000, 0.4711, -0.0203},
                    {0.2538, 0.0000, 0.0914, 0.0102, 0.5684}};
            int mtrALength = mtrA.length;

            double[] vecB = {0.7613, -0.8709, 3.2074, -1.8290, 2.9537};
            int vecBLength = vecB.length;;


            // Метод Гаусса
            System.out.println("|Метод Гаусса|");
            double[][] copyMtrA1 = copyMtr(mtrA, mtrALength);
            double[] copyVecB1 = copyVec(vecB, vecBLength);
            int numOfStepsInGaussMet = 0; // Кол-во итераций
            gaussMet(copyMtrA1, copyVecB1, numOfStepsInGaussMet, mtrALength, vecBLength);


            // Нахождение Х-ов
            double[] massX = findX(copyMtrA1, copyVecB1, mtrALength);
            int massXLength = massX.length;
            System.out.println("\nВектор решений X:");
            printVecWithoutRound(massX, massXLength);


            // Нахождение det(mtrA)
            System.out.println("\n\n|Нахождение det(mtrA)|");
            double detMtrA = detMtr(copyMtrA1, numOfStepsInGaussMet, mtrALength);
            System.out.println("det(mtrA) = " + detMtrA);


            // Нахождение обратной матрицы
            System.out.print("\n|Нахождение обратной матрицы|");
            double[][] singleMtr = new double[mtrALength][mtrALength]; // Создание единичной м-цы
            for (int i = 0; i < mtrALength; i++) {
                singleMtr[i][i] = 1;
            }
            double[][] copyMtrA2 = copyMtr(mtrA, mtrALength);
            double[][] reverseMtr = reverseMtr(copyMtrA2, singleMtr, mtrALength);


            // Нахождение числа обусловленности
            System.out.print("\n|Нахождения числа обусловленности|");
            double[][] copyMtrA3 = copyMtr(mtrA, mtrALength);
            System.out.println("\nЧисло обусловленности: " + (mtrNorm(copyMtrA3, mtrALength, "mtrA") * mtrNorm(reverseMtr, mtrALength, "reverseMtr")));


            // Нахождение вектора невязки
            System.out.print("\n|Нахождение невязки|");
            double[] copyVecB2 = copyVec(vecB, vecBLength);
            double[] massResidua = findResidua(copyMtrA3, copyVecB2, massX);
            System.out.print("\nВектор невязки:\n");
            printVecWithoutRound(massResidua, massXLength);


            // Матрица матрицы невязки
            System.out.print("\n\n|Матрица невязки|");
            double[][] copyMtrA4 = copyMtr(mtrA, mtrALength);
            printMtrWithoutRounding(calcResMtr(copyMtrA4, reverseMtr), mtrALength);
        }
        catch (ArithmeticException  e) {
            System.out.println("Ошибка: " + e.getMessage());
        }
    }


    // Глубокое копирование м-цы
    public static double[][] copyMtr(double[][] mtr, int mtrLength) {
        double[][] copyMtrA = new double[mtrLength][mtrLength];

        for (int i = 0; i < mtrLength; i++) {
            for (int j = 0; j < mtrLength; j++) {
                copyMtrA[i][j] = mtr[i][j];
            }
        }

        return copyMtrA;
    }


    // Глубокое копирование в-ра
    public static double[] copyVec(double[] vec, int vecLength) {
        double[] copyVecB = new double[vecLength];

        for (int i = 0; i < vecLength; i++) {
            copyVecB[i] = vec[i];
        }

        return copyVecB;
    }


    // Метод Гаусса
    public static void gaussMet (double[][] mtrA, double[] vecB, int numOfStepsInGaussMet, int mtrALength, int vecBLength) {
        System.out.print("До:");
        printMtrWithVec(mtrA, vecB, mtrALength, vecBLength);

        while (numOfStepsInGaussMet != mtrALength - 1) {
            // Поиск максимального элемента
            double maxEl = Math.abs(mtrA[numOfStepsInGaussMet][numOfStepsInGaussMet]);
            for (int i = numOfStepsInGaussMet; i < mtrALength; i++) {
                if (Math.abs(mtrA[i][numOfStepsInGaussMet]) > maxEl) {
                    maxEl = Math.abs(mtrA[i][numOfStepsInGaussMet]);
                }
            }


            // Перестановка строк
            for (int i = numOfStepsInGaussMet; i < mtrALength; i++) {
                if (Math.abs(mtrA[i][numOfStepsInGaussMet]) == maxEl) {
                    for (int j = 0; j < mtrALength; j++) {
                        // Для mtrA
                        double temp = mtrA[numOfStepsInGaussMet][j];
                        mtrA[numOfStepsInGaussMet][j] = mtrA[i][j];
                        mtrA[i][j] = temp;
                    }

                    // Для vecA
                    double temp = vecB[numOfStepsInGaussMet];
                    vecB[numOfStepsInGaussMet] = vecB[i];
                    vecB[i] = temp;
                }
            }


            // Высчитываем масштабирующие множители
            double[] massU = new double[mtrALength - 1];
            for (int i = numOfStepsInGaussMet + 1; i < mtrALength; i++) {
                massU[i - 1] = mtrA[i][numOfStepsInGaussMet] / mtrA[numOfStepsInGaussMet][numOfStepsInGaussMet];
            }


            // Преобразование матрицы
            for (int i = numOfStepsInGaussMet + 1; i < mtrALength; i++) {
                for (int j = numOfStepsInGaussMet; j < mtrALength; j++) {
                    mtrA[i][j] = mtrA[i][j] - (mtrA[numOfStepsInGaussMet][j] * massU[i - 1]); // Для mtrA
                }
                vecB[i] = vecB[i] - (vecB[numOfStepsInGaussMet] * massU[i - 1]); // Для vecA
            }

            numOfStepsInGaussMet++;
        }

        System.out.print("\nПосле:");
        printMtrWithVec(mtrA, vecB, mtrALength, vecBLength);

        System.out.println("\nКол-во шагов:\nk = " + (numOfStepsInGaussMet + 1));

        System.out.println("Матрица приведена к треугольному виду!");
    }


    // Нахождение X-ов
    public static double[] findX(double[][] mtrA, double[] vecB, int mtrALength) {
        double[] massX = new double[mtrALength];

        for (int i = mtrALength - 1; i >= 0; i--) {
            double sum = vecB[i];
            for (int j = mtrALength - 1; j > i; j--) {
                sum -= mtrA[i][j] * massX[j];
            }
            massX[i] = sum / mtrA[i][i];
        }

        return massX;
    }


    // Нахождение определителя м-цы (используя метод Гаусса)
    public static double detMtr(double[][] mtrA, int numOfStepsInGaussMet, int mtrALength) {
        double determinant = 1.0;
        for (int i = 0; i < mtrALength; i++) {
            determinant *= mtrA[i][i];
        }

        if (numOfStepsInGaussMet % 2 == 1) {
            return determinant * -1;
        }
        else {
            return determinant;
        }
    }


    // Нахождение обратной м-цы
    public static double[][] reverseMtr(double[][] copyMtr, double[][] singleMtr, int mtrALength) {
        System.out.print("\nДо:");
        printMtrWithMtr(copyMtr, singleMtr);

        // Преобразование в единичную матрицу
        for (int i = 0; i < mtrALength; i++) {
            double pivot = copyMtr[i][i];

            // Деление строки на pivot
            for (int j = 0; j < mtrALength; j++) {
                copyMtr[i][j] /= pivot;    // Для cMtr
                singleMtr[i][j] /= pivot;    // Для singleMtr
            }

            // Вычитание других строк для получения нулей под главной диагональю
            for (int k = 0; k < mtrALength; k++) {
                if (k != i) {
                    double factor = copyMtr[k][i];
                    for (int j = 0; j < mtrALength; j++) {
                        copyMtr[k][j] -= factor * copyMtr[i][j];    // Для cMtr
                        singleMtr[k][j] -= factor * singleMtr[i][j];    // Для singleMtr
                    }
                }
            }
        }

        System.out.print("\nПосле:");
        printMtrWithMtr(copyMtr, singleMtr);

        System.out.print("\nОбратная м-ца:");
        printMtr(singleMtr, singleMtr.length);

        return singleMtr;
    }


    // Норма м-цы
    public static double mtrNorm(double[][] mtrA, int mtrALength, String name) {
        double maxSum = 0.0;

        for (int j = 0; j < mtrALength; j++) {
            double columnSum = 0.0;
            for (int i = 0; i < mtrALength; i++) {
                columnSum += Math.abs(mtrA[i][j]);
            }
            maxSum = Math.max(maxSum, columnSum);
        }
        System.out.print("\n||" + name + "|| = " + maxSum);

        return maxSum;
    }


    // Вывод м-цы
    public static void printMtr(double[][] mtrA, int mtrALength) {
        System.out.println("\nВывод м-цы:");

        DecimalFormat decimalFormat = new DecimalFormat("#.####");

        for (int i = 0; i < mtrALength; i++) {
            for (int j = 0; j < mtrALength; j++) {
                System.out.print(decimalFormat.format(mtrA[i][j]));

                for (int k = 0; k < 15 - decimalFormat.format(mtrA[i][j]).length(); k++) {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }


    // Вывод м-цы и в-ра
    public static void printMtrWithVec(double[][] mtrA, double[] vecB, int mtrALength, int vecBLength) {
        System.out.println("\nВывод м-цы и в-ра:");

        DecimalFormat decimalFormat = new DecimalFormat("#.####");

        for (int i = 0; i < mtrALength; i++) {
            for (int j = 0; j < mtrALength; j++) {
                System.out.print(decimalFormat.format(mtrA[i][j]));

                for (int k = 0; k < 15 - decimalFormat.format(mtrA[i][j]).length(); k++) {
                    System.out.print(" ");
                }
            }
            System.out.println("|\t" + decimalFormat.format(vecB[i]));
        }
    }


    // Вывод м-цы и м-цы
    public static void printMtrWithMtr(double[][] mtr1, double[][] mtr2) {
        System.out.println("\nВывод м-цы и м-цы:");

        DecimalFormat decimalFormat = new DecimalFormat("#.####");

        for (int i = 0; i < mtr1.length; i++) {
            for (int j = 0; j < mtr1.length; j++) {
                System.out.print(decimalFormat.format(mtr1[i][j]));
                for (int k = 0; k < 15 - decimalFormat.format(mtr1[i][j]).length(); k++) {
                    System.out.print(" ");
                }
            }

            System.out.print("|\t");

            for (int j = 0; j < mtr2.length; j++) {
                System.out.print(decimalFormat.format(mtr2[i][j]));
                for (int k = 0; k < 15 - decimalFormat.format(mtr2[i][j]).length(); k++) {
                    System.out.print(" ");
                }
            }

            System.out.println();
        }
    }


    // Вывод вектора без округления
    public static void printVecWithoutRound(double[] vec, int vecLength) {
        System.out.println("Вывод в-ра:");

        for (int i = 0; i < vecLength; i++) {
            System.out.print(vec[i]);
            for (int j = 0; j < 30 - Double.toString(vec[i]).length(); j++) {
                System.out.print(" ");
            }
        }
    }


    // Вывод м-цы без округления
    public static void printMtrWithoutRounding(double[][] mtr, int mtrLength) {
        System.out.println("\nВывод м-цы");

        for (int i = 0; i < mtrLength; i++) {
            for (int j = 0; j < mtrLength; j++) {
                System.out.print(mtr[i][j]);
                for (int k = 0; k < 30 - Double.toString(mtr[i][j]).length(); k++) {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }


    // Нахождение в-ра невязки (r = A*x - b)
    public static double[] findResidua(double[][] mtrA, double[] vecB, double[] massX) {
        double[] newVec = new double[vecB.length];
        double sum = 0;

        for (int i = 0; i < mtrA.length; i++) {
            for (int j = 0; j < mtrA.length; j++) {
                sum += mtrA[i][j] * massX[j];
            }
            newVec[i] = sum;
            sum = 0.0;
        }

        for (int i = 0; i < vecB.length; i++) {
            newVec[i] = newVec[i] - vecB[i];
        }

        return newVec;
    }


    // Нахождение м-цы невязки (r = A*x - b)
    public static double[][] calcResMtr(double[][] mtrA, double[][] reverseMtr) {
        double[][] multMtr = new double[mtrA.length][reverseMtr[0].length];

        for (int i = 0; i < mtrA.length; i++) {
            for (int j = 0; j < reverseMtr[0].length; j++) {
                for (int k = 0; k < mtrA[0].length; k++) {
                    multMtr[i][j] += mtrA[i][k] * reverseMtr[k][j];
                }
            }
        }

        double[][] singleMtr = new double[mtrA.length][mtrA.length]; // Создание единичной м-цы
        for (int i = 0; i < mtrA.length; i++) {
            singleMtr[i][i] = 1;
        }

        double[][] subtraction = new double[multMtr.length][multMtr[0].length];
        for (int i = 0; i < multMtr.length; i++) {
            for (int j = 0; j < multMtr[0].length; j++) {
                subtraction[i][j] = multMtr[i][j] - singleMtr[i][j];
            }
        }

        return subtraction;
    }
}
