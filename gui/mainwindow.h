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
class QCloseEvent;
class QMenu;
class QAction;
class QActionGroup;
class QToolButton;

class MainWindow : public QWidget {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void startHelper();
    void stopHelper();
    void writeControls();
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
    void setScaleAction(uint32_t mode);
    void setQualityAction(uint32_t quality);

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
    QPushButton* browseRunnerBtn = nullptr;
    QComboBox* runnerCombo = nullptr;
    QLineEdit* runnerPathEdit = nullptr;
    QLabel* statusLabel = nullptr;
    QProcess* statusProcess = nullptr;
    bool helperRunning = false;

    QCheckBox* enabledBox = nullptr;
    QSpinBox* passesSpin = nullptr;
    QComboBox* presetCombo = nullptr;
    QDoubleSpinBox* intensitySpin = nullptr;
    QDoubleSpinBox* localToneSpin = nullptr;
    QDoubleSpinBox* localStructureSpin = nullptr;
    QDoubleSpinBox* skinStructureSpin = nullptr;
    QDoubleSpinBox* sharpnessSpin = nullptr;

    QToolButton* settingsBtn = nullptr;
    QMenu* settingsMenu = nullptr;
    QAction* mvecAction = nullptr;
    QMenu* scaleMenu = nullptr;
    QMenu* qualityMenu = nullptr;
    QActionGroup* scaleGroup = nullptr;
    QActionGroup* qualityGroup = nullptr;
    QAction* scaleNormalizedAction = nullptr;
    QAction* scalePixelsAction = nullptr;
    QAction* scaleUv01Action = nullptr;
    QAction* qualityFastAction = nullptr;
    QAction* qualityBalancedAction = nullptr;
    QAction* qualityQualityAction = nullptr;
};