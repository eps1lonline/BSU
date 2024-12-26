#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void printEmployees(string filename) {
	ifstream in(filename, ios::binary);

	int num;
	char name[10];
	double hours;

	while (!in.eof()) {
		in.read((char*)&num, sizeof(int));
		in.read(name, sizeof(char) * 10);
		in.read((char*)&hours, sizeof(double));

		cout << "Num:   \t" << num << " "
			<< "Name:  \t" << name << " "
			<< "Hours: \t" << hours << endl;
	}
}

void printFileContent(string filename) {
	ifstream in(filename);
	string buffer;

	while (getline(in, buffer))
		cout << buffer << endl;
}

int main() {
	STARTUPINFO startInf;
	PROCESS_INFORMATION processInf;

	ZeroMemory(&startInf, sizeof(STARTUPINFO));
	startInf.cb = sizeof(STARTUPINFO);

	string inFilename;
	cout << "Enter binary file name: ";
	cin >> inFilename;

	int n;
	cout << "Enter number of employees: ";
	cin >> n;

	stringstream ss;
	ss << "Creator.exe " << inFilename << ", " << n;

	char* args = (char*)malloc(ss.str().length() + 1);
	strcpy(args, ss.str().c_str());

	if (!CreateProcess(NULL, TEXT(args), NULL, NULL, FALSE, 0, NULL, NULL, &startInf, &processInf)) {
		printf("CreateProcess failed (%d).", GetLastError());
		return -1;
	}

	WaitForSingleObject(processInf.hProcess, INFINITE);

	CloseHandle(processInf.hProcess);
	CloseHandle(processInf.hThread);

	cout << "\nEmployees:\n";
	printEmployees(inFilename);

	string repFilename;
	cout << "\Input report filename: ";
	cin >> repFilename;

	double price;
	cout << "Input price: ";
	cin >> price;

	ss = stringstream();
	ss << "Reporter.exe " << inFilename << " " << repFilename << " " << price;

	args = (char*)malloc(ss.str().length() + 1);
	strcpy(args, ss.str().c_str());

	if (!CreateProcess(NULL, TEXT(args), NULL, NULL, FALSE, 0, NULL, NULL, &startInf, &processInf))
		return -1;

	WaitForSingleObject(processInf.hProcess, INFINITE);

	CloseHandle(processInf.hProcess);
	CloseHandle(processInf.hThread);

	cout << "\nReport:\n";
	printFileContent(repFilename);

	system("pause");

	return 0;
}