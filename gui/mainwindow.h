#pragma once
#include <QWidget>
#include <QString>
#include "../common/shm_protocol.h"

class QProcess;
class QPushButton;
class QCheckBox;
class QSpinBox;
class QDoubleSpinBox;
class QLabel;
class QTimer;
class QComboBox;
class QLineEdit;
class ShmBinder;

class MainWindow : public QWidget {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void startHelper();
    void stopHelper();
    void updateStatus();

private:
    QString findProjectDir() const;
    QString findHelperCli() const;
    QString configPath() const;
    QString defaultShmPath() const;
    QString defaultLogPath() const;
    void loadConfig();
    void saveConfig();
    void populateRunners();
    void applyRunnerSelection(int index);
    bool ensureShm();
    QWidget* buildSettings();

    QString projectDir;
    QString helperCliPath;
    QString configFilePath;
    QString runnerType;
    QString runnerPath;
    QString binariesPath;
    QString logPath;
    QString dxvkVendor;
    QString dxvkDevice;
    QString shmPath;
    void* shmBase = nullptr;
    ShmHeader* hdr = nullptr;

    QProcess* helper = nullptr;
    QTimer* statusTimer = nullptr;

    QPushButton* startBtn = nullptr;
    QPushButton* stopBtn = nullptr;
    QPushButton* passBtn = nullptr;
    QPushButton* captureBtn = nullptr;
    QPushButton* browseRunnerBtn = nullptr;
    QComboBox* runnerCombo = nullptr;
    QLineEdit* runnerPathEdit = nullptr;
    QLabel* statusLabel = nullptr;
    QLabel* costLabel = nullptr;
    QLabel* whitePointLabel = nullptr;
    QSpinBox* captureFrames = nullptr;
    QProcess* statusProcess = nullptr;
    bool helperRunning = false;

    ShmBinder* binder = nullptr;
};
