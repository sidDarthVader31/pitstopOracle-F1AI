
import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn } from "typeorm";

@Entity("driver")
export class Driver {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  full_name: string;

  @Column({ nullable: true })
  nationality: string;

  @Column({ type: "date", nullable: true })
  date_of_birth: Date;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
