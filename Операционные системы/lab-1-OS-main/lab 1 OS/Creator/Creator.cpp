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
};

int main() {
	string filename;
	cin >> filename;

	int counter;
	cin >> counter;

	ofstream out(filename, ios::binary);

	cout << "\nInput information about employees (ID, name, hours):\n";

	for (size_t i = 0; i < counter; i++) {
		Employee emp;
		cin >> emp.num >> emp.name >> emp.hours;

		out << emp;
	}

	return 0;
}