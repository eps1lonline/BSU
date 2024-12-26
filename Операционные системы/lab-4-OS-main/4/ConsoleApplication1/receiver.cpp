#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <Windows.h>

using namespace std;

int main() {
	string fname;
	cout << "Enter filename:\n";
	cin >> fname;

	int count;
	cout << "Enter buffer size:\n";
	cin >> count;

	int procCount;
	cout << "Enter count of senders:\n";
	cin >> procCount;

	STARTUPINFO* si = new STARTUPINFO[procCount];
	PROCESS_INFORMATION* pi = new PROCESS_INFORMATION[procCount];

	ZeroMemory(si, sizeof(STARTUPINFO) * procCount);
	ZeroMemory(pi, sizeof(PROCESS_INFORMATION) * procCount);

	for (int i = 0; i < procCount; i++) {
		si[i].cb = sizeof(si);
	}

	HANDLE mutex = CreateMutex(NULL, FALSE, "GlobalFileCommunication");
	if (mutex == INVALID_HANDLE_VALUE) {
		printf("Error (%d)", GetLastError());
	}

	HANDLE semaphore = CreateSemaphore(NULL, count, count, "BufferSizeSemaphore");
	if (semaphore == INVALID_HANDLE_VALUE) {
		printf("Error (%d)", GetLastError());
	}

	fstream in;
	HANDLE file = CreateFile(TEXT(fname.c_str()), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
	HANDLE* proc = new HANDLE[procCount];
	string command = ("Sender.exe " + fname).c_str();
	char* commandCstr = new char[command.length() + 1];

	strcpy(commandCstr, command.c_str());

	for (int i = 0; i < procCount; i++) {
		if (!CreateProcess(NULL, TEXT(commandCstr), NULL, NULL, TRUE, CREATE_NEW_CONSOLE, NULL, NULL, &si[i], &pi[i])) {
			printf("Error (%d)", GetLastError());
		}
	}

	TCHAR pwd[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, pwd);

	cout << "Commands:\n"
		<< "exit\n"
		<< "read\n";

	char buffer[20];
	while (command != "exit") {
		cin >> command;
		if (command == "read") {

			bool failed;
			do {
				WaitForSingleObject(mutex, INFINITE);
				in.open(fname, ios::binary | ios::in);
				in.read(buffer, 20);

				failed = in.fail();
				string contents = "";
				while (!in.fail()) {
					contents += (char)in.get();
				}
				if (!contents.empty()) {
					contents.erase(contents.end() - 1);
				}
				in.close();

				in.open(fname, ios::binary | ios::out | ios::trunc);
				in.write(contents.c_str(), contents.length());
				in.flush();

				in.close();

				ReleaseMutex(mutex);
			} while (failed);
			ReleaseSemaphore(semaphore, 1, NULL);

			cout << buffer << endl;
		}
	}

	for (int i = 0; i < procCount; i++) {
		CloseHandle(proc[i]);
	}

	CloseHandle(file);
	CloseHandle(mutex);
	CloseHandle(semaphore);

	return 0;
}