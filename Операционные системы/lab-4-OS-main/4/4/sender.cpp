#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <Windows.h>

using namespace std;

void sendMessage(string filename, string message, HANDLE& mutex, HANDLE& semaphore);

int main(int argc, char** argv) {
	string filename = argv[1];

	HANDLE mutex = OpenMutex(MUTEX_ALL_ACCESS, TRUE, "GlobalFileCommunication");
	if (mutex == NULL) {
		printf("Error (%d)", GetLastError());
	}

	HANDLE semaphore = OpenSemaphore(SEMAPHORE_ALL_ACCESS, TRUE, "BufferSizeSemaphore");
	if (semaphore == NULL) {
		printf("Error (%d)", GetLastError());
	}

	cout << "\nCommands:\n"
		<< "exit\n"
		<< "read\n";

	string command;
	while (command != "exit") {
		cin >> command;
		if (command == "send") {
			cin >> command;
			sendMessage(filename, command, mutex, semaphore);
		}
	}

	CloseHandle(mutex);
	CloseHandle(semaphore);

	return 0;
}

void sendMessage(string filename, string message, HANDLE& mutex, HANDLE& semaphore) {
	if (message.length() >= 20) {
		cout << "Message length exceeds 20\n";
		return;
	}

	WaitForSingleObject(semaphore, INFINITE);
	WaitForSingleObject(mutex, INFINITE);

	char target[20];
	strcpy(target, message.c_str());
	fstream out(filename, ios::binary | ios::out | ios::app);

	out.write(target, 20);
	out.flush();
	out.close();

	ReleaseMutex(mutex);
}