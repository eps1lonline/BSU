#include <iostream>
#include <fstream>

using namespace std;

struct Employee {
	int num;
	char name[10];
	double hours;

	friend ostream& operator<<(ostream& out, Employee& emp) {
		out.write((char*)&emp.num, sizeof(int));
		out.write(emp.name, sizeof(char) * 10);
		out.write((char*)&emp.hours, sizeof(double));

		return out;
	}

	friend istream& operator>>(istream& in, Employee& emp) {
		in.read((char*)&emp.num, sizeof(int));
		in.read(emp.name, sizeof(char) * 10);
		in.read((char*)&emp.hours, sizeof(double));

		return in;
	}
};

int main() {
	string inFilename;
	cin >> inFilename;

	string repFilename;
	cin >> repFilename;

	int price;
	cin >> price;

	ifstream in(inFilename);
	ofstream out(repFilename);

	out << "Report for file " << inFilename << endl;

	Employee emp;
	while (!in.eof()) {
		in >> emp;

		out << "Num:   \t" << emp.num << " "
			<< "Name:  \t" << emp.name << " "
			<< "Hours: \t" << emp.hours << " "
			<< "Wages: \t" << price * emp.hours << endl;
	}

	return 0;
}