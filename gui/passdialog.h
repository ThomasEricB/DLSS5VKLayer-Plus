#pragma once
#include <QDialog>
#include <array>
#include "../common/shm_protocol.h"

class QCheckBox;
class QDoubleSpinBox;
class QTabWidget;

class PassDialog : public QDialog {
    Q_OBJECT
public:
    explicit PassDialog(ShmHeader* hdr, QWidget* parent = nullptr);

private slots:
    void writePass(int pass);

private:
    struct Row {
        QCheckBox* enabled = nullptr;
        QDoubleSpinBox* intensity = nullptr;
        QDoubleSpinBox* localTone = nullptr;
        QDoubleSpinBox* localStructure = nullptr;
        QDoubleSpinBox* skinStructure = nullptr;
        QDoubleSpinBox* sharpness = nullptr;
    };

    ShmHeader* hdr = nullptr;
    QTabWidget* tabs = nullptr;
    std::array<Row, kMaxPasses> rows;
};